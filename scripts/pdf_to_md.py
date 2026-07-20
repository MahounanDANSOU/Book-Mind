#!/usr/bin/env python3
"""
Convertit un PDF en Markdown, en conservant la numérotation de page
pour permettre un chunking avec métadonnée `page` (voir architecture BookMind).

Usage:
    python pdf_to_md.py chemin/vers/fichier.pdf [dossier_sortie]

Sortie: un fichier .md avec des marqueurs "## Page N" avant le texte
de chaque page. Ces marqueurs sont exploités par ingest.py pour
associer chaque chunk à sa page d'origine.
"""
import sys
import re
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Erreur : pymupdf non installé. Lancer : pip install pymupdf")
    sys.exit(1)


def nettoyer_texte_page(texte: str) -> str:
    """Supprime espaces multiples et lignes vides excessives."""
    texte = re.sub(r"[ \t]+", " ", texte)
    texte = re.sub(r"\n{3,}", "\n\n", texte)
    return texte.strip()


def detecter_lignes_repetees(pages_texte: list[str], seuil: float = 0.6) -> set[str]:
    """
    Détecte les lignes courtes (en-têtes/pieds de page) qui se répètent
    sur une grande proportion des pages, pour les retirer ensuite.
    """
    from collections import Counter
    compteur = Counter()
    for texte in pages_texte:
        lignes = {l.strip() for l in texte.split("\n") if 0 < len(l.strip()) < 80}
        compteur.update(lignes)
    n_pages = max(len(pages_texte), 1)
    return {ligne for ligne, n in compteur.items() if n / n_pages >= seuil}


def pdf_vers_markdown(chemin_pdf: str, dossier_sortie: str = "data/raw") -> str:
    chemin = Path(chemin_pdf)
    if not chemin.exists():
        raise FileNotFoundError(f"Introuvable : {chemin}")

    doc = fitz.open(chemin)
    pages_brutes: list[str] = [str(page.get_text("text")) for page in doc]
    doc.close()

    if not any(p.strip() for p in pages_brutes):
        raise ValueError(
            f"Aucun texte extrait de {chemin.name} — "
            "probablement un PDF scanné (image). Nécessite de l'OCR, non couvert ici."
        )

    lignes_a_retirer = detecter_lignes_repetees(pages_brutes)

    blocs_md = [f"# {chemin.stem}\n"]
    for i, texte in enumerate(pages_brutes, start=1):
        lignes_filtrees = [
            l for l in texte.split("\n") if l.strip() not in lignes_a_retirer
        ]
        texte_propre = nettoyer_texte_page("\n".join(lignes_filtrees))
        if texte_propre:
            blocs_md.append(f"## Page {i}\n\n{texte_propre}\n")

    contenu_md = "\n".join(blocs_md)

    dossier = Path(dossier_sortie)
    dossier.mkdir(parents=True, exist_ok=True)
    chemin_sortie = dossier / f"{chemin.stem}.md"
    chemin_sortie.write_text(contenu_md, encoding="utf-8")
    return str(chemin_sortie)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_md.py fichier.pdf [dossier_sortie]")
        sys.exit(1)
    sortie = sys.argv[2] if len(sys.argv) > 2 else "data/raw"
    resultat = pdf_vers_markdown(sys.argv[1], sortie)
    print(f"OK -> {resultat}")