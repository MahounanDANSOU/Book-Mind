"""Évaluation de la recherche : Recall@k + MRR sur eval.jsonl
(k = top_k du config, le nombre de résultats que rechercher() retourne).

Seuil bloquant (feuille de route Phase 5) : Recall@k >= 80%.
"""
import json

from config import CONFIG
from search import rechercher, IndexAbsentError


def charger_eval(eval_path: str) -> list[dict]:
    questions = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne or ligne.startswith("//"):
                continue
            questions.append(json.loads(ligne))
    return questions


def evaluer_question(item: dict) -> dict:
    resultats = rechercher(item["question"])
    titres_trouves = [r["titre"] for r in resultats]

    doc_attendu = item["doc_attendu"]
    # tolère avec ou sans extension .txt dans eval.jsonl
    doc_attendu_stem = doc_attendu[:-4] if doc_attendu.endswith(".txt") else doc_attendu

    trouve_en_rang = None
    for i, titre in enumerate(titres_trouves):
        if titre == doc_attendu_stem or titre == doc_attendu:
            trouve_en_rang = i + 1
            break

    mot_cle = item.get("mot_cle_extrait")
    mot_cle_present = False
    if mot_cle:
        # espaces normalisés : le texte des chunks contient des retours à la ligne
        mot_cle_present = any(
            mot_cle.lower() in " ".join(r["texte"].split()).lower() for r in resultats)

    return {
        "question": item["question"],
        "doc_attendu": doc_attendu,
        "trouve": trouve_en_rang is not None,
        "rang": trouve_en_rang,
        "mot_cle_present": mot_cle_present if mot_cle else None,
        "reciprocal_rank": 1.0 / trouve_en_rang if trouve_en_rang else 0.0,
    }


def evaluer() -> dict:
    eval_path = CONFIG["eval_path"]

    items = charger_eval(eval_path)
    if not items:
        print(f"Aucune question dans {eval_path}")
        return {}

    resultats = [evaluer_question(item) for item in items]

    k = CONFIG["top_k"]
    recall_at_k = sum(1 for r in resultats if r["trouve"]) / len(resultats)
    mrr = sum(r["reciprocal_rank"] for r in resultats) / len(resultats)

    print(f"Questions évaluées : {len(resultats)}")
    print(f"Recall@{k} : {recall_at_k:.1%}")
    print(f"MRR      : {mrr:.3f}")
    avec_mot_cle = [r for r in resultats if r["mot_cle_present"] is not None]
    if avec_mot_cle:
        taux = sum(1 for r in avec_mot_cle if r["mot_cle_present"]) / len(avec_mot_cle)
        print(f"Mot-clé retrouvé dans les extraits : {taux:.1%} "
              f"({len(avec_mot_cle)} questions avec mot-clé)")
    print()

    for r in resultats:
        statut = "OK " if r["trouve"] else "ECHEC"
        rang = f"rang {r['rang']}" if r["rang"] else "non trouvé"
        print(f"[{statut}] {r['question'][:60]:60s} -> {r['doc_attendu']} ({rang})")

    seuil = 0.80
    if recall_at_k >= seuil:
        print(f"\nSeuil atteint (Recall@{k} >= {seuil:.0%}). Passage Phase 6 autorisé.")
    else:
        print(f"\nSeuil NON atteint (Recall@{k} < {seuil:.0%}). Ajuster chunk_size / nettoyage / modèle avant Phase 6.")

    return {"recall_at_k": recall_at_k, "k": k, "mrr": mrr, "details": resultats}


if __name__ == "__main__":
    import sys
    try:
        evaluer()
    except IndexAbsentError as e:
        print(f"ERREUR : {e}")
        sys.exit(1)
