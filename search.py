#!/usr/bin/env python3
"""
Recherche hybride BookMind : fusionne recherche lexicale (FTS5/BM25) et
recherche sémantique (embeddings) via Reciprocal Rank Fusion (RRF).
"""
import json
import re
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np

from config import CONFIG
from embeddings import encoder, description_modele


class IndexAbsentError(RuntimeError):
    """L'index documentaire (base SQLite + embeddings) n'existe pas encore,
    ou n'est pas compatible avec la configuration actuelle."""


_MSG_INDEX_ABSENT = (
    "L'index documentaire n'existe pas encore. Placez des documents dans "
    f"{CONFIG['raw_dir']} puis lancez « python ingest.py » (ou start_app.bat)."
)

_emb_cache: "np.ndarray | None" = None
_ids_cache: "np.ndarray | None" = None


def _connexion() -> sqlite3.Connection:
    """Connexion à la base — refuse de créer silencieusement une base vide
    (sqlite3.connect créerait le fichier, puis toute requête échouerait en
    « no such table », un message incompréhensible pour l'utilisateur)."""
    if not Path(CONFIG["db_path"]).exists():
        raise IndexAbsentError(_MSG_INDEX_ABSENT)
    return sqlite3.connect(CONFIG["db_path"])


def _verifier_compatibilite_index() -> None:
    """L'index doit avoir été encodé avec le modèle d'embeddings de la config —
    sinon la recherche comparerait des vecteurs incompatibles et retournerait
    des passages aberrants SANS aucune erreur. Les index d'avant l'introduction
    des métadonnées sont réputés construits avec l'ancien modèle par défaut."""
    meta_path = Path(CONFIG["db_path"]).parent / "index_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:  # index historique, d'avant index_meta.json
        meta = {"embedding_backend": "sentence-transformers",
                "embedding_model": "intfloat/multilingual-e5-small"}
    actuel = description_modele()
    if (meta.get("embedding_model") != actuel["embedding_model"]
            or meta.get("embedding_backend") != actuel["embedding_backend"]):
        raise IndexAbsentError(
            f"L'index a été construit avec le modèle d'embeddings "
            f"« {meta.get('embedding_model')} » mais la configuration utilise "
            f"« {actuel['embedding_model']} ». Relancez « python ingest.py » "
            f"(ou le bouton Recharger l'index) pour reconstruire l'index.")


def _charger_ressources() -> tuple["np.ndarray", "np.ndarray"]:
    global _emb_cache, _ids_cache
    if _emb_cache is None or _ids_cache is None:
        _verifier_compatibilite_index()
        try:
            _emb_cache = np.load(CONFIG["emb_path"])
            _ids_cache = np.load(CONFIG["ids_path"])
        except FileNotFoundError:
            raise IndexAbsentError(_MSG_INDEX_ABSENT) from None
    return _emb_cache, _ids_cache


def recharger_index():
    """Force le rechargement des embeddings/ids depuis le disque au prochain appel
    (à utiliser après une ré-ingestion, pour éviter de redémarrer le serveur)."""
    global _emb_cache, _ids_cache
    _emb_cache = None
    _ids_cache = None


# Mots vides français/anglais : les laisser dans la requête BM25 (jointe en OR)
# fait remonter les chunks les plus verbeux au lieu des plus pertinents.
_MOTS_VIDES = frozenset("""
au aux avec ce ces cette dans de des du elle elles en et eux il ils je la le les
leur leurs lui ma mais me mes moi mon ne nos notre nous on ou où par pas pour qu
que quel quelle quelles quels qui sa se ses son sur ta te tes toi ton tu un une
vos votre vous y a à c d j l m n s t est sont était sera comme plus très tout
tous toute toutes autre autres aussi bien être avoir fait faire dit dire cela ça
donc alors quand si non oui the of and to in is are was for that this it as be
by on or an at from with what which who how why does do
el los las que un una por para con no su al lo como mas pero sus ya este esta
entre cuando muy sin sobre tambien hasta hay donde quien desde todo nos otros
esto antes ellos cual cuales
""".split())


def _nettoyer_requete_fts(question: str) -> str:
    """Échappe la requête pour FTS5 : garde les mots porteurs de sens, les joint en OR.
    Les mots vides ne sont retirés que s'il reste au moins un mot plein."""
    mots = re.findall(r"\w+", question, re.UNICODE)
    mots = [m for m in mots if len(m) > 1]
    pleins = [m for m in mots if m.lower() not in _MOTS_VIDES]
    if pleins:
        mots = pleins
    if not mots:
        return '""'
    return " OR ".join(f'"{m}"' for m in mots)


def _recherche_lexicale(question: str, conn: sqlite3.Connection, k: int | None = None):
    """Classement BM25 sur TOUT le corpus (k=None = pas de limite) : la base
    documentaire ne contient que quelques dizaines de milliers de chunks, la classer
    en entier est instantané — plafonner artificiellement risquerait d'exclure un
    livre pertinent dont le meilleur passage n'est pas dans le tout premier lot."""
    requete = _nettoyer_requete_fts(question)
    sql = "SELECT rowid, bm25(chunks_fts) as score FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score"
    params: list[str | int] = [requete]
    if k is not None:
        sql += " LIMIT ?"
        params.append(k)
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        # requête FTS5 invalide malgré le nettoyage → on laisse la recherche
        # sémantique seule faire le travail plutôt que de planter
        return []
    # bm25() renvoie un score où PLUS PETIT = meilleur ; on convertit en rang croissant
    return [row[0] for row in rows]  # déjà trié du meilleur au moins bon


def _recherche_semantique(question: str, k: int | None = None):
    """Similarité cosinus contre TOUT le corpus (k=None = classement complet)."""
    emb, ids = _charger_ressources()
    q_vec = encoder([question], type_texte="query")[0]
    scores = emb @ q_vec  # cosinus (vecteurs déjà normalisés) — calculé sur tous les chunks
    if k is not None and k < scores.shape[0]:
        # sélection partielle O(N) puis tri des k retenus seulement
        idx = np.argpartition(-scores, k)[:k]
        top_idx = idx[np.argsort(-scores[idx])]
    else:
        top_idx = np.argsort(-scores)
    return [int(ids[i]) for i in top_idx]


def _normaliser(texte: str) -> str:
    """Minuscules, accents retirés, tout ce qui n'est pas alphanumérique → espace."""
    texte = unicodedata.normalize("NFD", texte.lower())
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", texte).strip()


_MOTS_VIDES_NORM = frozenset(_normaliser(m) for m in _MOTS_VIDES)

# Suffixes techniques des noms de fichiers de livres (ex. '..._interieur_20250708063033_1607')
_SUFFIXE_TITRE_RE = re.compile(r"_interieur.*$|_\d{8,}.*$")


def _mots_titre(titre: str) -> list[str]:
    """Mots porteurs de sens du titre d'un document (suffixes techniques et mots vides retirés)."""
    base = _SUFFIXE_TITRE_RE.sub("", titre)
    return [m for m in _normaliser(base).split()
            if len(m) > 1 and m not in _MOTS_VIDES_NORM]


def _boosts_titre_par_doc(conn: sqlite3.Connection, question: str) -> dict[int, float]:
    """Détecte les livres dont le titre est (presque) cité dans la question.
    Retourne {doc_id: ratio} pour les titres dont au moins 60 % des mots porteurs
    figurent dans la question — ex. « En finir avec le passé » doit faire remonter
    le livre du même nom, même si son contenu ne matche pas mot à mot."""
    mots_question = set(_normaliser(question).split())
    if not mots_question:
        return {}
    boosts = {}
    rows = conn.execute(
        "SELECT id, titre FROM documents WHERE titre NOT LIKE '%.metadata'"
    ).fetchall()
    for doc_id, titre in rows:
        mots = _mots_titre(titre)
        if len(mots) < 2:
            continue  # titre trop court pour être discriminant
        ratio = sum(1 for m in mots if m in mots_question) / len(mots)
        if ratio >= 0.6:
            boosts[doc_id] = ratio
    return boosts


def _fusion_rrf(listes_rangs: list[list[int]], k_rrf: int = 60) -> dict[int, float]:
    scores = {}
    for liste in listes_rangs:
        for rang, chunk_id in enumerate(liste):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k_rrf + rang + 1)
    return scores


def _charger_chunks(conn: sqlite3.Connection, chunk_ids: list[int]) -> list[dict]:
    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"""SELECT c.id, c.texte, c.position, c.page, d.titre
            FROM chunks c JOIN documents d ON c.doc_id = d.id
            WHERE c.id IN ({placeholders})""",
        chunk_ids,
    ).fetchall()
    par_id = {r[0]: {"id": r[0], "texte": r[1], "position": r[2],
                      "page": r[3], "titre": r[4]} for r in rows}
    return [par_id[cid] for cid in chunk_ids if cid in par_id]


def rechercher(question: str, top_k: int | None = None) -> list[dict]:
    """Retourne les meilleurs chunks (texte + source) pour une question, en
    comparant la question à TOUT le contenu de TOUS les livres (BM25 + sémantique
    classent l'intégralité du corpus, sans plafond de candidats avant fusion)."""
    top_k = top_k or CONFIG["top_k"]
    conn = _connexion()

    lex = _recherche_lexicale(question, conn)
    sem = _recherche_semantique(question)
    scores = _fusion_rrf([lex, sem])

    # Si la question cite le titre d'un livre, les chunks de ce livre sont favorisés
    if scores:
        boosts = _boosts_titre_par_doc(conn, question)
        if boosts:
            placeholders = ",".join("?" * len(scores))
            rows = conn.execute(
                f"SELECT id, doc_id FROM chunks WHERE id IN ({placeholders})",
                list(scores.keys()),
            ).fetchall()
            for cid, did in rows:
                if did in boosts:
                    scores[cid] *= 1.0 + boosts[did]

    meilleurs_ids = [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]

    resultats = _charger_chunks(conn, meilleurs_ids)
    conn.close()
    return resultats


def rechercher_documents(question: str, top_k: int = 10) -> list[dict]:
    """Agrège les scores de chunks par document — pour 'quels documents parlent de X'.

    Le score d'un document = somme de ses 3 meilleurs chunks (pas de tous) : sommer
    tous les chunks favorisait mécaniquement les livres longs, qui accumulent des
    dizaines de correspondances médiocres et écrasent un livre court très pertinent.
    Un livre dont le titre est cité dans la question reçoit un fort bonus, même si
    son contenu ne ressort pas dans les chunks. Comme rechercher(), classe TOUT le
    corpus (pas de plafond de candidats) : aucun livre n'est exclu d'office."""
    conn = _connexion()
    lex = _recherche_lexicale(question, conn)
    sem = _recherche_semantique(question)
    scores = _fusion_rrf([lex, sem])
    boosts = _boosts_titre_par_doc(conn, question)

    if not scores and not boosts:
        conn.close()
        return []

    chunks_par_doc: dict[int, list[float]] = {}
    titres = {}
    if scores:
        placeholders = ",".join("?" * len(scores))
        rows = conn.execute(
            f"SELECT c.id, c.doc_id, d.titre FROM chunks c JOIN documents d ON c.doc_id = d.id "
            f"WHERE c.id IN ({placeholders})",
            list(scores.keys()),
        ).fetchall()
        for chunk_id, doc_id, titre in rows:
            chunks_par_doc.setdefault(doc_id, []).append(scores[chunk_id])
            titres[doc_id] = titre

    scores_par_doc = {doc_id: sum(sorted(vals, reverse=True)[:3])
                      for doc_id, vals in chunks_par_doc.items()}

    # Bonus titre : 0.15 × ratio dépasse le score contenu maximal (~0.10), donc un
    # livre explicitement nommé dans la question sort en tête même sans chunk retrouvé.
    for doc_id, ratio in boosts.items():
        scores_par_doc[doc_id] = scores_par_doc.get(doc_id, 0.0) + 0.15 * ratio
        if doc_id not in titres:
            titres[doc_id] = conn.execute(
                "SELECT titre FROM documents WHERE id = ?", (doc_id,)).fetchone()[0]
    conn.close()

    classement = sorted(scores_par_doc.items(), key=lambda x: -x[1])[:top_k]
    return [{"doc_id": doc_id, "titre": titres[doc_id], "score": score}
            for doc_id, score in classement]


def lister_livres() -> list[dict]:
    """Liste les livres de la base avec leur plage de pages et leur nombre de chapitres
    détectés (pour un sélecteur d'UI). Exclut les fiches .metadata et ne garde que les
    documents réellement paginés."""
    conn = _connexion()
    rows = conn.execute(
        """SELECT d.titre, MIN(c.page), MAX(c.page), COUNT(c.id),
                  (SELECT COUNT(*) FROM chapitres ch WHERE ch.doc_id = d.id)
           FROM documents d JOIN chunks c ON c.doc_id = d.id
           WHERE d.titre NOT LIKE '%.metadata'
             AND c.page IS NOT NULL
           GROUP BY d.id
           HAVING COUNT(c.id) > 1
           ORDER BY d.titre"""
    ).fetchall()
    conn.close()
    return [{"titre": t, "page_min": pmin, "page_max": pmax, "nb_chunks": n, "nb_chapitres": nc}
            for t, pmin, pmax, n, nc in rows]


def lister_chapitres(titre_livre: str) -> list[dict]:
    """Sommaire détecté d'un livre : liste ordonnée de {titre, page_debut, page_fin}.
    page_fin de chaque chapitre = page de début du suivant moins un (ou la dernière
    page du livre pour le tout dernier chapitre). Retourne [] si aucun sommaire n'a
    été détecté à l'ingestion (l'app doit alors proposer une plage de pages manuelle)."""
    conn = _connexion()
    doc = conn.execute("SELECT id FROM documents WHERE titre = ?", (titre_livre,)).fetchone()
    if not doc:
        conn.close()
        return []
    doc_id = doc[0]
    chapitres = conn.execute(
        "SELECT titre, page_debut FROM chapitres WHERE doc_id = ? ORDER BY position",
        (doc_id,),
    ).fetchall()
    if not chapitres:
        conn.close()
        return []
    page_max = conn.execute(
        "SELECT MAX(page) FROM chunks WHERE doc_id = ?", (doc_id,)
    ).fetchone()[0]
    conn.close()

    resultat = []
    for i, (titre, page_debut) in enumerate(chapitres):
        if i + 1 < len(chapitres):
            page_fin = chapitres[i + 1][1] - 1
        else:
            page_fin = page_max
        resultat.append({"titre": titre, "page_debut": page_debut,
                         "page_fin": max(page_fin, page_debut)})
    return resultat


def chunks_par_plage(titre: str, page_debut: int, page_fin: int) -> list[dict]:
    """Retourne, dans l'ordre du livre, tous les chunks d'un livre entre deux pages incluses.
    Sert à traiter un chapitre entier (défini par sa plage de pages), et non un top-k."""
    conn = _connexion()
    rows = conn.execute(
        """SELECT c.texte, c.position, c.page, d.titre
           FROM chunks c JOIN documents d ON c.doc_id = d.id
           WHERE d.titre = ? AND c.page >= ? AND c.page <= ?
           ORDER BY c.position""",
        (titre, page_debut, page_fin),
    ).fetchall()
    conn.close()
    return [{"texte": r[0], "position": r[1], "page": r[2], "titre": r[3]} for r in rows]


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "test"
    for r in rechercher(q):
        print(f"[{r['titre']} — position {r['position']} page {r['page']}] {r['texte'][:100]}...")