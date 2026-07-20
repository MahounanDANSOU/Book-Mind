"""Chargement centralisé de config.yaml.

- Les chemins sont résolus par rapport à la racine du projet (dossier de ce fichier),
  pas au dossier courant — les scripts fonctionnent quel que soit l'endroit d'où ils
  sont lancés.
- Valide les valeurs critiques (un chevauchement >= taille de chunk ferait boucler
  le découpage sans jamais avancer).
- Force le mode hors-ligne Hugging Face par défaut (aucun appel réseau une fois
  le modèle d'embeddings en cache local).
"""
import os
from pathlib import Path

import yaml

os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")

RACINE = Path(__file__).resolve().parent

CONFIG = yaml.safe_load((RACINE / "config.yaml").read_text(encoding="utf-8"))

# Chemins du config relatifs à la racine du projet — toute clé *_path / *_dir,
# présente ou future, est résolue (une liste figée oublierait les nouvelles clés)
for _cle, _val in list(CONFIG.items()):
    if isinstance(_val, str) and (_cle.endswith("_path") or _cle.endswith("_dir")):
        CONFIG[_cle] = str(RACINE / _val)

if CONFIG["chunk_overlap_mots"] >= CONFIG["chunk_size_mots"]:
    raise ValueError(
        "config.yaml : chunk_overlap_mots doit être strictement inférieur à "
        "chunk_size_mots (sinon le découpage en chunks ne peut pas avancer)."
    )
