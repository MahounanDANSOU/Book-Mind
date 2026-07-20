"""Test rapide de l'encodage vectoriel (backend et modèle du config.yaml)."""
from embeddings import encoder, description_modele

print("Modèle actif :", description_modele())
v = encoder(["test d'encodage"], type_texte="query")
print("Vecteur :", v.shape, "— norme :", float((v[0] ** 2).sum() ** 0.5))
