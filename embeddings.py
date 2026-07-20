"""Encodage vectoriel centralisé — point d'entrée unique pour ingest.py et search.py.

Deux backends, choisis par `embedding_backend` dans config.yaml :
- "ollama" : le modèle d'embeddings (ex. bge-m3) est servi par Ollama, comme le LLM.
  Meilleure qualité multilingue, aucun téléchargement Hugging Face nécessaire
  (important sur ce réseau où huggingface.co est bloqué), et aucun modèle
  supplémentaire chargé dans le processus Python.
- "sentence-transformers" : modèle Hugging Face déjà en cache local (ex. e5-small).

Tous les vecteurs retournés sont float32 et normalisés L2 (le produit scalaire
est alors directement le cosinus).
"""
import numpy as np
import requests

from config import CONFIG


class EmbeddingError(RuntimeError):
    """Échec d'encodage — message déjà rédigé pour l'utilisateur."""


_BACKEND = str(CONFIG.get("embedding_backend", "sentence-transformers"))
_MODELE = str(CONFIG["embedding_model"])

_st_modele = None  # cache du modèle sentence-transformers (backend HF uniquement)


def _url_embed_ollama() -> str:
    # ollama_url pointe sur /api/generate — on en dérive /api/embed
    return CONFIG["ollama_url"].rsplit("/api/", 1)[0] + "/api/embed"


# Troncature de sécurité : un texte anormalement long (ligne de pointillés d'un
# sommaire, artefact OCR) déborderait la fenêtre du modèle → Ollama répond 400
# pour TOUT le lot. 8000 caractères ≈ bien au-delà d'un chunk normal (~1600).
_MAX_CAR_TEXTE = 8000


def _encoder_ollama(textes: list[str]) -> np.ndarray:
    textes = [t[:_MAX_CAR_TEXTE] for t in textes]
    try:
        r = requests.post(
            _url_embed_ollama(),
            json={"model": _MODELE, "input": textes, "keep_alive": "30m"},
            timeout=600,
        )
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.ConnectionError:
        raise EmbeddingError(
            "Impossible de contacter Ollama pour calculer les embeddings — "
            "vérifie qu'Ollama est lancé.") from None
    except requests.exceptions.HTTPError:
        try:
            detail = r.json().get("error", r.text)
        except ValueError:
            detail = r.text
        raise EmbeddingError(
            f"Ollama a refusé le calcul d'embeddings ({r.status_code}) : "
            f"{str(detail)[:300]}") from None
    except requests.exceptions.RequestException as e:
        raise EmbeddingError(f"Échec du calcul d'embeddings via Ollama : {e}") from None
    vecteurs = data.get("embeddings")
    if not vecteurs:
        raise EmbeddingError(
            f"Ollama n'a pas renvoyé d'embeddings (modèle « {_MODELE} » absent ? "
            f"Lancez : ollama pull {_MODELE})")
    return np.asarray(vecteurs, dtype="float32")


def _encoder_st(textes: list[str], type_texte: str) -> np.ndarray:
    global _st_modele
    if _st_modele is None:
        from sentence_transformers import SentenceTransformer
        _st_modele = SentenceTransformer(_MODELE, local_files_only=True)
    # Les modèles e5 exigent un préfixe de rôle ("query:" / "passage:")
    if "e5" in _MODELE:
        prefixe = "query: " if type_texte == "query" else "passage: "
        textes = [prefixe + t for t in textes]
    return _st_modele.encode(textes, convert_to_numpy=True,
                             normalize_embeddings=True,
                             show_progress_bar=len(textes) > 100).astype("float32")


def encoder(textes: list[str], type_texte: str = "passage",
            lot: int = 64, progression: bool = False) -> np.ndarray:
    """Encode une liste de textes en vecteurs float32 L2-normalisés.

    type_texte : "passage" (indexation) ou "query" (question) — certains modèles
    (e5) distinguent les deux rôles.
    lot : taille des lots envoyés à Ollama (le backend HF gère ses lots lui-même).
    """
    if not textes:
        return np.zeros((0, 0), dtype="float32")
    if _BACKEND != "ollama":
        return _encoder_st(textes, type_texte)

    morceaux = []
    for i in range(0, len(textes), lot):
        morceaux.append(_encoder_ollama(textes[i:i + lot]))
        if progression and len(textes) > lot:
            fait = min(i + lot, len(textes))
            print(f"\r  Embeddings : {fait}/{len(textes)}", end="", flush=True)
    if progression and len(textes) > lot:
        print()
    vecteurs = np.vstack(morceaux)
    # normalisation L2 (Ollama ne garantit pas des vecteurs normalisés)
    normes = np.linalg.norm(vecteurs, axis=1, keepdims=True)
    normes[normes == 0] = 1.0
    return vecteurs / normes


def description_modele() -> dict:
    """Identité du modèle d'embeddings actif — stockée dans les métadonnées de
    l'index à l'ingestion, et vérifiée au chargement par la recherche (un index
    encodé avec un autre modèle produirait des résultats aberrants sans erreur)."""
    return {"embedding_backend": _BACKEND, "embedding_model": _MODELE}
