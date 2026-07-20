#!/usr/bin/env python3
"""Génère un jeu d'évaluation (data/eval.jsonl) à partir de la base documentaire.

Pour chaque livre échantillonné, prend un chunk au milieu du livre et demande au
LLM local d'écrire une question dont la réponse se trouve dans ce passage, plus
une expression exacte copiée du passage. Le document attendu est connu par
construction (c'est celui du chunk) — ce qui permet ensuite de mesurer
objectivement Recall@k et MRR avec evaluate.py.

Usage : python scripts/generer_eval.py [nb_questions]   (défaut : 12)
Les questions existantes de eval.jsonl sont conservées (ajout à la suite).
"""
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from generate import _appel_ollama, OllamaError

GEN_TEMPLATE = """Voici un passage tiré d'un livre.

PASSAGE :
{texte}

Écris UNE question précise en français qu'un lecteur pourrait poser et dont la \
réponse se trouve dans ce passage. La question ne doit pas mentionner le mot \
« passage » ni « texte » — elle doit se suffire à elle-même. \
Elle doit reprendre les éléments DISTINCTIFS du passage (noms propres, lieux, \
chiffres, expressions marquantes) et non des généralités qui pourraient \
s'appliquer à n'importe quel livre.
Puis recopie une expression EXACTE de 2 à 5 mots, présente telle quelle dans le \
passage, qui porte la réponse.

Réponds STRICTEMENT sur deux lignes, sans rien d'autre :
QUESTION: <la question>
EXPRESSION: <l'expression exacte copiée du passage>"""

_REPONSE_RE = re.compile(
    r"QUESTION\s*:\s*(?P<question>.+?)\s*\n\s*EXPRESSION\s*:\s*(?P<expression>.+)",
    re.IGNORECASE | re.DOTALL,
)


def generer(nb_questions: int = 12) -> None:
    # sqlite3.connect créerait silencieusement une base vide (qui fausserait ensuite
    # le test d'existence de search._connexion) — on vérifie d'abord.
    if not Path(CONFIG["db_path"]).exists():
        print("Base documentaire absente — lancez d'abord : python ingest.py")
        return
    conn = sqlite3.connect(CONFIG["db_path"])
    docs = conn.execute(
        """SELECT d.id, d.titre, COUNT(c.id) AS nb
           FROM documents d JOIN chunks c ON c.doc_id = d.id
           WHERE d.titre NOT LIKE '%.metadata'
           GROUP BY d.id HAVING nb >= 10"""
    ).fetchall()
    if not docs:
        print("Base vide ou trop petite — lancez d'abord python ingest.py")
        return

    random.shuffle(docs)
    docs = docs[:nb_questions]

    eval_path = Path(CONFIG["eval_path"])
    nb_ok = 0
    with eval_path.open("a", encoding="utf-8") as f:
        for doc_id, titre, nb in docs:
            # chunk au milieu du livre : évite préface, sommaire et pages de garde
            row = conn.execute(
                "SELECT texte FROM chunks WHERE doc_id = ? ORDER BY position "
                "LIMIT 1 OFFSET ?",
                (doc_id, nb // 2),
            ).fetchone()
            if not row or len(row[0]) < 300:
                continue
            texte = row[0][:2500]

            print(f"  Génération d'une question pour : {titre[:60]}...")
            try:
                brut = _appel_ollama(GEN_TEMPLATE.format(texte=texte),
                                     num_predict=150, num_ctx=4096,
                                     timeout=300, temperature=0.3)
            except OllamaError as e:
                print(f"ARRÊT : {e}")
                break

            m = _REPONSE_RE.search(brut)
            if not m:
                print(f"    (format non respecté, passage ignoré)")
                continue
            question = " ".join(m.group("question").split())
            expression = " ".join(m.group("expression").split()).strip('"« »')
            # l'expression doit réellement figurer dans le passage, sinon inutilisable.
            # Comparaison en espaces normalisés des DEUX côtés : le texte source
            # contient des retours à la ligne, l'expression recopiée n'en a plus.
            texte_norm = " ".join(texte.split()).lower()
            if expression.lower() not in texte_norm:
                print(f"    (expression « {expression[:40]} » absente du passage, ignoré)")
                continue

            f.write(json.dumps({"question": question, "doc_attendu": titre,
                                "mot_cle_extrait": expression},
                               ensure_ascii=False) + "\n")
            f.flush()
            nb_ok += 1
            print(f"    OK : {question[:70]}")

    conn.close()
    print(f"\n{nb_ok} question(s) ajoutée(s) à {eval_path}")
    print("Mesurez maintenant la qualité de la recherche : python evaluate.py")


if __name__ == "__main__":
    nb = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    generer(nb)
