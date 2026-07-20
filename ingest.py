#!/usr/bin/env python3
"""
Ingestion BookMind : lit tous les .txt et .md de data/raw/, découpe en chunks,
stocke dans SQLite (+ index FTS5) et calcule les embeddings (numpy).

Usage : python ingest.py
Relancer à chaque ajout/modification de document (ré-ingestion complète).
"""
import json
import os
import re
import sqlite3
import bisect
from pathlib import Path

import numpy as np

from config import CONFIG

# PaddleOCR pour les pages sans couche texte (GPU auto-détecté si paddlepaddle-gpu installé).
# Chargement lazy à la première utilisation pour ne pas ralentir si OCR désactivé.
_OCR_MODEL = None

def _get_ocr_model():
    """Initialise le pipeline PaddleOCR (lazy). Retourne None si indisponible."""
    global _OCR_MODEL
    if _OCR_MODEL is None:
        try:
            # Le projet tourne avec HF_HUB_OFFLINE=1 ; or PaddleOCR 3.x télécharge ses
            # modèles via Hugging Face par défaut. On force la source BOS (serveurs
            # Paddle) pour que le premier téléchargement du modèle OCR fonctionne.
            os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
            from paddleocr import PaddleOCR
            # API PaddleOCR 3.x : la langue vient du config ('fr' par défaut) ; les modules
            # d'orientation/déformation sont désactivés (inutiles sur des pages de livre,
            # et coûteux en RAM/temps).
            # enable_mkldnn=False : l'accélération CPU oneDNN de paddlepaddle plante sur
            # CE modèle (PP-OCRv6) avec "ConvertPirAttribute2RuntimeAttribute not
            # support [pir::ArrayAttribute<DoubleAttribute>]" — bug d'incompatibilité
            # entre le nouvel exécuteur PIR de Paddle et le noyau oneDNN. Sans ce
            # paramètre, TOUTE page nécessitant l'OCR échouait silencieusement (texte
            # vide), quel que soit le modèle de langue choisi.
            _OCR_MODEL = PaddleOCR(
                lang=str(CONFIG.get("ocr_langue", "fr")),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )
        except ModuleNotFoundError as e:
            print(f"  Attention : module '{e.name}' manquant — lancez setup_ocr.bat "
                  f"pour installer l'OCR. OCR desactive.")
            _OCR_MODEL = False  # marker pour ne pas réessayer
        except Exception as e:
            print(f"  Attention : PaddleOCR non disponible ({e}). OCR desactive.")
            _OCR_MODEL = False
    return _OCR_MODEL if _OCR_MODEL else None

PAGE_MARKER = re.compile(r"^## Page (\d+)\s*$", re.MULTILINE)

MANIFEST_PATH = Path(CONFIG["db_path"]).parent / "manifest.json"


def _etat_sources() -> dict:
    """Photographie des fichiers sources (nom → [taille, mtime]) pour détecter les changements."""
    raw_dir = Path(CONFIG["raw_dir"])
    etat = {}
    for f in sorted(raw_dir.iterdir()) if raw_dir.exists() else []:
        if f.suffix.lower() not in (".txt", ".md", ".pdf"):
            continue
        if f.name.startswith("_converti_") or ".metadata" in f.name:
            continue
        s = f.stat()
        etat[f.name] = [s.st_size, int(s.st_mtime)]
    return etat


def ingestion_necessaire() -> bool:
    """True si des fichiers ont été ajoutés/modifiés/supprimés depuis la dernière ingestion."""
    if not MANIFEST_PATH.exists():
        return True
    try:
        ancien = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return True
    return ancien != _etat_sources()


def extraire_pages(texte: str) -> tuple[str, list[tuple[int, int]]]:
    """
    Si le texte contient des marqueurs '## Page N' (issus de la conversion PDF),
    les retire et retourne (texte_sans_marqueurs, [(offset_char, num_page), ...]).
    Sinon retourne (texte, []) — pas de métadonnée page (cas des .txt).
    """
    matches = list(PAGE_MARKER.finditer(texte))
    if not matches:
        return texte, []

    morceaux = []
    frontieres = []
    offset = 0
    for i, m in enumerate(matches):
        debut_contenu = m.end()
        fin_contenu = matches[i + 1].start() if i + 1 < len(matches) else len(texte)
        contenu = texte[debut_contenu:fin_contenu]
        morceaux.append(contenu)
        frontieres.append((offset, int(m.group(1))))
        offset += len(contenu)

    return "".join(morceaux), frontieres


def page_pour_offset(frontieres: list[tuple[int, int]], offset: int) -> int | None:
    if not frontieres:
        return None
    positions = [f[0] for f in frontieres]
    idx = bisect.bisect_right(positions, offset) - 1
    idx = max(idx, 0)
    return frontieres[idx][1]


_FIN_PHRASE_RE = re.compile(r"[.!?…]+[»\"')\]]*\s+|\n{2,}")


def _spans_phrases(texte: str) -> list[tuple[int, int]]:
    """Découpe le texte en phrases : liste de (début, fin) en offsets de caractères.
    Fins de phrase : ponctuation forte suivie d'un blanc, ou paragraphe (double saut)."""
    spans, debut = [], 0
    for m in _FIN_PHRASE_RE.finditer(texte):
        if m.end() > debut and texte[debut:m.end()].strip():
            spans.append((debut, m.end()))
        debut = m.end()
    if debut < len(texte) and texte[debut:].strip():
        spans.append((debut, len(texte)))
    return spans


def _redecouper_dur(texte: str, deb: int, fin: int, taille_mots: int):
    """Redécoupe une « phrase » démesurée : par mots, et en dernier recours par
    tranches de caractères — une ligne de pointillés de sommaire
    (« Introduction..........45 ») est un seul « mot » de plusieurs milliers de
    caractères, insécable par espaces, qui ferait déborder la fenêtre du modèle
    d'embeddings."""
    budget_car = taille_mots * 10
    courant_deb = None
    nb_mots = car = 0
    derniere_fin = deb
    for m in re.finditer(r"\S+", texte[deb:fin]):
        m_deb, m_fin = deb + m.start(), deb + m.end()
        long_mot = m_fin - m_deb
        if long_mot > budget_car:
            # mot géant sans espaces : tranches brutes de caractères
            if courant_deb is not None:
                yield (courant_deb, derniere_fin)
                courant_deb, nb_mots, car = None, 0, 0
            for p in range(m_deb, m_fin, budget_car):
                yield (p, min(p + budget_car, m_fin))
            derniere_fin = m_fin
            continue
        if courant_deb is None:
            courant_deb, nb_mots, car = m_deb, 0, 0
        nb_mots += 1
        car += long_mot + 1
        derniere_fin = m_fin
        if nb_mots >= taille_mots or car >= budget_car:
            yield (courant_deb, m_fin)
            courant_deb, nb_mots, car = None, 0, 0
    if courant_deb is not None and derniere_fin > courant_deb:
        yield (courant_deb, derniere_fin)


def chunker(texte: str, frontieres: list[tuple[int, int]],
            taille_mots: int, chevauchement_mots: int):
    """Découpe le texte en chunks ALIGNÉS SUR LES PHRASES : chaque chunk regroupe des
    phrases entières jusqu'à ~taille_mots, avec un chevauchement d'au moins
    chevauchement_mots (en phrases entières) entre chunks consécutifs.

    Couper au milieu d'une phrase (ancien découpage au compteur de mots) dégradait
    à la fois l'embedding (débuts/fins de phrase orphelins) et la lisibilité des
    passages affichés comme sources."""
    budget_car = taille_mots * 10  # plafond de sécurité en caractères par chunk
    spans = []  # (début, fin, nb_mots)
    for deb, fin in _spans_phrases(texte):
        nb = len(texte[deb:fin].split())
        if nb <= taille_mots and (fin - deb) <= budget_car:
            spans.append((deb, fin, nb))
        else:
            for d2, f2 in _redecouper_dur(texte, deb, fin, taille_mots):
                spans.append((d2, f2, len(texte[d2:f2].split())))
    if not spans:
        return

    position_chunk = 0
    i = 0
    while i < len(spans):
        j, nb_mots, nb_car = i, 0, 0
        while j < len(spans) and (nb_mots == 0
                                  or (nb_mots + spans[j][2] <= taille_mots
                                      and nb_car + (spans[j][1] - spans[j][0]) <= budget_car)):
            nb_mots += spans[j][2]
            nb_car += spans[j][1] - spans[j][0]
            j += 1
        char_start, char_end = spans[i][0], spans[j - 1][1]
        texte_chunk = texte[char_start:char_end].strip()
        if texte_chunk:
            yield {
                "position": position_chunk,
                "texte": texte_chunk,
                "char_start": char_start,
                "char_end": char_end,
                "page": page_pour_offset(frontieres, char_start),
            }
            position_chunk += 1
        if j >= len(spans):
            break
        # chevauchement : reprendre quelques phrases en arrière (>= chevauchement_mots)
        recul, k = 0, j
        while k > i + 1 and recul < chevauchement_mots:
            k -= 1
            recul += spans[k][2]
        i = k


def _page_est_blanche(page) -> bool:
    """Détecte une page entièrement blanche (pas de contenu à lire) via un rendu
    basse résolution — quasi instantané, contre 9-14s pour un appel OCR complet.
    Les livres imprimés contiennent souvent des pages de mise en page totalement
    vides (alignement recto/verso) : sur cette bibliothèque, elles représentaient
    jusqu'à 100% des pages 'sans texte natif' d'un livre — les OCRiser pour rien
    faisait passer une ré-ingestion de ~40 minutes à plusieurs heures."""
    import fitz
    pix = page.get_pixmap(matrix=fitz.Matrix(0.3, 0.3), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    return bool((arr < 250).sum() == 0)


def _texte_page_ocr(page) -> str:
    """OCR d'une page via PaddleOCR.
    Retourne '' si l'OCR échoue, si PaddleOCR n'est pas installé, ou si la page
    est blanche (vérifié avant l'appel OCR — voir _page_est_blanche)."""
    try:
        import fitz

        if _page_est_blanche(page):
            return ""

        ocr = _get_ocr_model()
        if ocr is None:
            return ""

        # Page PDF → tableau numpy (2x zoom pour meilleure OCR). pix.samples est en
        # RGB ; PaddleOCR (basé OpenCV) attend du BGR → inversion des canaux.
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img = np.ascontiguousarray(img[:, :, ::-1])

        # API PaddleOCR 3.x : predict() renvoie un résultat par image, avec les
        # lignes reconnues dans rec_texts (objet dict-like ou attribut selon la version)
        lignes = []
        for res in ocr.predict(img):
            textes = res.get("rec_texts") if hasattr(res, "get") else None
            if textes is None:
                textes = getattr(res, "rec_texts", None)
            lignes.extend(textes or [])
        return "\n".join(lignes)
    except Exception as e:
        print(f"    (échec OCR sur une page : {e})")
        return ""


def extraire_texte_pdf_avec_pages(chemin_pdf: Path) -> str | None:
    """Convertit un PDF en texte avec marqueurs '## Page N', ou None si aucun texte récupérable.

    - Fait taire les avertissements MuPDF non fatals (ex. 'Screen annotations').
    - Robuste page par page : une page défectueuse n'interrompt pas tout le document.
    - Si config['ocr_fallback'] est vrai, applique PaddleOCR aux pages sans couche texte.
    """
    import fitz
    fitz.TOOLS.mupdf_display_errors(False)  # coupe le bruit stderr des annotations/objets exotiques

    ocr_actif = bool(CONFIG.get("ocr_fallback", False))

    try:
        doc = fitz.open(chemin_pdf)
    except Exception as e:
        print(f"  ATTENTION : impossible d'ouvrir {chemin_pdf.name} ({e}). Ignoré.")
        return None

    pages = []
    nb_ocr = 0
    for page in doc:
        try:
            texte = str(page.get_text("text"))
        except Exception:
            texte = ""
        if not texte.strip() and ocr_actif:
            texte = _texte_page_ocr(page)
            if texte.strip():
                nb_ocr += 1
        pages.append(texte)
    doc.close()

    if nb_ocr:
        print(f"    (OCR appliqué à {nb_ocr} page(s) sans couche texte)")
    if not any(p.strip() for p in pages):
        return None

    # Nettoyage : en-têtes/pieds répétés sur la plupart des pages (titre du livre,
    # nom de chapitre courant) et numéros de page isolés — du bruit présent dans
    # chaque chunk, qui pollue BM25 et les embeddings.
    parasites = _lignes_parasites(pages)
    blocs = []
    for i, texte in enumerate(pages, start=1):
        lignes = [l for l in texte.split("\n")
                  if l.strip() not in parasites and not l.strip().isdigit()]
        texte_propre = "\n".join(lignes).strip()
        if texte_propre:
            blocs.append(f"## Page {i}\n\n{texte_propre}\n")
    return "\n".join(blocs)


def _lignes_parasites(pages: list[str], seuil: float = 0.6) -> set[str]:
    """Lignes courtes identiques présentes sur >= seuil des pages non vides :
    ce sont les en-têtes/pieds de page répétés, à retirer avant indexation."""
    from collections import Counter
    non_vides = [p for p in pages if p.strip()]
    compteur = Counter()
    for texte in non_vides:
        lignes = {l.strip() for l in texte.split("\n") if 0 < len(l.strip()) < 80}
        compteur.update(lignes)
    n = max(len(non_vides), 1)
    return {ligne for ligne, c in compteur.items() if c / n >= seuil}


_CHAPITRE_RE = re.compile(r"^chapitre\s+\d+", re.IGNORECASE)


def _sommaire_natif(doc) -> list[dict]:
    """Sommaire via les signets (bookmarks) intégrés au PDF. Rejette les faux sommaires
    de type 'un signet par diapositive' (PDF issus de PowerPoint)."""
    toc = doc.get_toc()
    if len(toc) < 2:
        return []
    titres = [t for _, t, _ in toc]
    ratio_slide = sum(1 for t in titres if t.strip().lower().startswith(("diapo", "slide"))) / len(titres)
    if ratio_slide > 0.5:
        return []
    return [{"titre": t.strip(), "page_debut": p} for _, t, p in toc if p and p >= 1]


def _sommaire_typographique(doc) -> list[dict]:
    """Détecte les chapitres via la taille de police : cherche la taille de titre pour
    laquelle des lignes suivent le motif 'Chapitre N', et regroupe le titre qui suit
    (même taille, même page) comme suite du titre du chapitre."""
    lignes_par_taille_page: dict[float, dict[int, list[str]]] = {}
    for num_page, page in enumerate(doc, start=1):
        d = page.get_text("dict")
        for bloc in d["blocks"]:
            for ligne in bloc.get("lines", []):
                texte = "".join(s["text"] for s in ligne["spans"]).strip()
                if not texte:
                    continue
                tailles = [round(s["size"], 1) for s in ligne["spans"] if s["text"].strip()]
                if not tailles:
                    continue
                taille = max(tailles)
                lignes_par_taille_page.setdefault(taille, {}).setdefault(num_page, []).append(texte)

    # Choisit la plus grande taille de police pour laquelle le motif 'Chapitre N' apparaît
    # au moins 2 fois — évite de confondre un titre de couverture avec un vrai repère de chapitre.
    for taille in sorted(lignes_par_taille_page.keys(), reverse=True):
        pages = lignes_par_taille_page[taille]
        nb_matches = sum(1 for lignes in pages.values() if lignes and _CHAPITRE_RE.match(lignes[0]))
        if nb_matches >= 2:
            sommaire = []
            for num_page in sorted(pages.keys()):
                lignes = pages[num_page]
                if _CHAPITRE_RE.match(lignes[0]):
                    titre = " ".join(lignes[:4]).strip()
                    sommaire.append({"titre": titre, "page_debut": num_page})
            return sommaire
    return []


def extraire_sommaire(doc) -> list[dict]:
    """Sommaire d'un livre (liste ordonnée de {titre, page_debut}), en essayant
    d'abord les signets intégrés au PDF, puis une détection typographique (taille de
    police + motif 'Chapitre N'). Retourne [] si rien de fiable n'est détecté —
    dans ce cas l'app proposera une plage de pages manuelle en dernier recours."""
    sommaire = _sommaire_natif(doc)
    if sommaire:
        return sommaire
    return _sommaire_typographique(doc)


def _remplacer_atomique(tmp: Path, final: Path, essais: int = 20) -> None:
    """os.replace avec retries : sous Windows le remplacement échoue si l'app tient
    encore une connexion SQLite ouverte une fraction de seconde."""
    import time
    for i in range(essais):
        try:
            os.replace(tmp, final)
            return
        except PermissionError:
            if i == essais - 1:
                raise
            time.sleep(0.5)


def init_db(chemin_db: str) -> sqlite3.Connection:
    Path(chemin_db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(chemin_db)
    conn.executescript("""
        DROP TABLE IF EXISTS chunks_fts;
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS chapitres;
        DROP TABLE IF EXISTS documents;

        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            titre TEXT NOT NULL,
            chemin TEXT NOT NULL
        );

        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            position INTEGER NOT NULL,
            texte TEXT NOT NULL,
            char_start INTEGER,
            char_end INTEGER,
            page INTEGER,
            FOREIGN KEY (doc_id) REFERENCES documents(id)
        );

        CREATE TABLE chapitres (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            position INTEGER NOT NULL,
            titre TEXT NOT NULL,
            page_debut INTEGER NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents(id)
        );

        CREATE VIRTUAL TABLE chunks_fts USING fts5(texte, content='chunks', content_rowid='id');
    """)
    conn.commit()
    return conn


META_PATH = Path(CONFIG["db_path"]).parent / "index_meta.json"


def main():
    raw_dir = Path(CONFIG["raw_dir"])
    fichiers_txt_md = sorted(
        [f for f in list(raw_dir.glob("*.txt")) + list(raw_dir.glob("*.md"))
         if not f.name.startswith("_converti_") and ".metadata" not in f.name]
    )
    fichiers_pdf = sorted(raw_dir.glob("*.pdf"))
    if not fichiers_txt_md and not fichiers_pdf:
        print(f"Aucun fichier .txt/.md/.pdf trouvé dans {raw_dir}/")
        return

    # Construction dans des fichiers temporaires, bascule atomique à la toute fin :
    # l'app peut continuer à servir l'ANCIEN index (base + embeddings cohérents entre
    # eux) pendant toute la ré-ingestion. Reconstruire en place donnait, pendant
    # plusieurs minutes, une base à moitié remplie interrogée avec les anciens
    # embeddings → résultats aberrants sans aucune erreur.
    db_final = Path(CONFIG["db_path"])
    emb_final = Path(CONFIG["emb_path"])
    ids_final = Path(CONFIG["ids_path"])
    db_tmp = db_final.with_name("~" + db_final.name)
    emb_tmp = emb_final.with_name("~" + emb_final.name)
    ids_tmp = ids_final.with_name("~" + ids_final.name)
    if db_tmp.exists():
        db_tmp.unlink()

    conn = init_db(str(db_tmp))
    cur = conn.cursor()

    tous_chunks_texte = []  # pour les embeddings, dans l'ordre d'insertion
    tous_chunks_ids = []

    def traiter_document(titre: str, chemin_str: str, texte_brut: str) -> int:
        texte, frontieres = extraire_pages(texte_brut)
        cur.execute("INSERT INTO documents (titre, chemin) VALUES (?, ?)",
                    (titre, chemin_str))
        doc_id = cur.lastrowid
        for chunk in chunker(texte, frontieres,
                              CONFIG["chunk_size_mots"], CONFIG["chunk_overlap_mots"]):
            cur.execute(
                "INSERT INTO chunks (doc_id, position, texte, char_start, char_end, page) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (doc_id, chunk["position"], chunk["texte"],
                 chunk["char_start"], chunk["char_end"], chunk["page"]),
            )
            chunk_id = cur.lastrowid
            cur.execute("INSERT INTO chunks_fts (rowid, texte) VALUES (?, ?)",
                        (chunk_id, chunk["texte"]))
            tous_chunks_texte.append(chunk["texte"])
            tous_chunks_ids.append(chunk_id)
        return doc_id

    for fichier in fichiers_txt_md:
        texte_brut = fichier.read_text(encoding="utf-8", errors="ignore")
        traiter_document(fichier.stem, str(fichier), texte_brut)
        print(f"  {fichier.name} -> {len(tous_chunks_ids)} chunks cumulés")

    for fichier in fichiers_pdf:
        print(f"  Conversion PDF : {fichier.name}...")
        texte_avec_pages = extraire_texte_pdf_avec_pages(fichier)
        if texte_avec_pages is None:
            print(f"  ATTENTION : {fichier.name} ne contient aucun texte extractible "
                  f"(probablement un PDF scanné). Ignoré — nécessite de l'OCR.")
            continue
        # Sauvegarde pour inspection qualité (important pour PDF issus de PowerPoint)
        chemin_md_debug = raw_dir / f"_converti_{fichier.stem}.md"
        chemin_md_debug.write_text(texte_avec_pages, encoding="utf-8")
        doc_id = traiter_document(fichier.stem, str(fichier), texte_avec_pages)
        print(f"  {fichier.name} -> {len(tous_chunks_ids)} chunks cumulés "
              f"(texte converti visible dans {chemin_md_debug.name})")

        import fitz
        fitz.TOOLS.mupdf_display_errors(False)
        try:
            doc_fitz = fitz.open(fichier)
            sommaire = extraire_sommaire(doc_fitz)
            doc_fitz.close()
        except Exception:
            sommaire = []
        for position, chap in enumerate(sommaire):
            cur.execute(
                "INSERT INTO chapitres (doc_id, position, titre, page_debut) VALUES (?, ?, ?, ?)",
                (doc_id, position, chap["titre"], chap["page_debut"]),
            )
        if sommaire:
            print(f"    Sommaire détecté : {len(sommaire)} chapitre(s)")

    conn.commit()
    conn.close()

    if not tous_chunks_texte:
        print("Aucun chunk produit — vérifier le contenu des fichiers.")
        db_tmp.unlink(missing_ok=True)  # l'ancien index reste intact
        return

    from embeddings import encoder, description_modele
    print(f"Encodage de {len(tous_chunks_texte)} chunks avec {CONFIG['embedding_model']}...")
    embeddings = encoder(tous_chunks_texte, type_texte="passage", progression=True)

    np.save(emb_tmp, embeddings)
    np.save(ids_tmp, np.array(tous_chunks_ids, dtype="int64"))
    meta_tmp = META_PATH.with_name("~" + META_PATH.name)
    meta_tmp.write_text(json.dumps({**description_modele(),
                                    "dim": int(embeddings.shape[1]),
                                    "nb_chunks": len(tous_chunks_ids)},
                                   ensure_ascii=False, indent=1), encoding="utf-8")

    # Bascule atomique : les fichiers de l'index sont remplacés ensemble, seulement
    # maintenant que tout a réussi. Un crash avant ce point laisse l'ancien index
    # complet et cohérent.
    _remplacer_atomique(db_tmp, db_final)
    _remplacer_atomique(emb_tmp, emb_final)
    _remplacer_atomique(ids_tmp, ids_final)
    _remplacer_atomique(meta_tmp, META_PATH)

    # Manifeste sauvegardé uniquement après une ingestion complète réussie
    MANIFEST_PATH.write_text(json.dumps(_etat_sources(), ensure_ascii=False, indent=1),
                             encoding="utf-8")

    print(f"Terminé : {len(fichiers_txt_md) + len(fichiers_pdf)} documents traités, "
          f"{len(tous_chunks_texte)} chunks, embeddings sauvegardés dans {CONFIG['emb_path']}")


if __name__ == "__main__":
    import sys
    if "--si-nouveau" in sys.argv:
        if not ingestion_necessaire():
            print("Base documentaire à jour — aucune ingestion nécessaire.")
            sys.exit(0)
        print("Changements détectés dans data/raw/ — ingestion...")
    main()