# BookMind — Recensement des améliorations possibles

État des lieux au 19/07/2026, classé par impact ressenti sur la qualité des résultats,
avec l'effort estimé. Les éléments déjà en place ne sont pas répétés (voir README).

## 🥇 Fort impact, faisable maintenant

1. **Streaming de la réponse dans le chat** — aujourd'hui on attend 30-60 s devant un
   spinner ; avec le streaming Ollama (`stream: true` + `st.write_stream`), la réponse
   s'écrit mot à mot dès les premières secondes, comme ChatGPT. Ne change rien à la
   qualité mais transforme la perception de vitesse. *Effort : moyen (generate.py + app.py).*

2. **Jeu d'évaluation plus large et plus réaliste** — 10 questions auto-générées, c'est
   trop peu pour piloter les réglages (en cours d'élargissement à ~25). Le complément
   idéal : 10-15 **vraies questions** que tu te poses réellement, ajoutées à la main dans
   `data/eval.jsonl` — c'est le meilleur thermomètre possible. *Effort : 30 min de ton temps.*

3. **Reformulation de la question par un petit LLM avant recherche** — les questions
   floues ou conversationnelles retrouvent mal ; qwen3:0.6b (déjà installé, rapide même
   sur CPU) peut générer 2-3 reformulations/mots-clés fusionnés ensuite par RRF.
   Améliore le rappel sur les questions imprécises. *Effort : moyen.*

4. **Réactiver l'analyse par source dans le chat** — la fonction `analyser_extrait()`
   existe (le LLM explique ce que CHAQUE source dit sur la question) mais n'est plus
   branchée dans l'interface depuis le passage au chat. Un bouton « analyser cette
   source » dans l'expander suffirait. *Effort : faible.*

## 🥈 Fort impact, bloqué ou coûteux

5. **Reranker cross-encoder** (ex. `bge-reranker-v2-m3`) — LE levier classique manquant :
   un modèle qui relit les 30 meilleurs candidats et les reclasse finement. **Bloqué** :
   huggingface.co est inaccessible depuis ce réseau (SSL coupé). Contournements possibles :
   - télécharger le modèle depuis un **partage de connexion mobile** puis le copier dans
     le cache local (`~/.cache/huggingface`) — 10 minutes de manip ;
   - ou vérifier dans l'antivirus/pare-feu ce qui bloque les connexions OCSP/TLS
     (le même blocage touche pypi.org et la vérification de certificats GitHub).
   *Effort : faible une fois le modèle récupéré.*

6. **Une machine avec GPU Nvidia** — tout s'accélère d'un facteur 10-30 sans changer une
   ligne de code (Ollama détecte le GPU automatiquement) : réponses en 2-5 s, ingestion
   en minutes. C'est LA limite structurelle actuelle. *Effort : matériel.*

7. **Ingestion incrémentale** — aujourd'hui, ajouter 1 livre ré-encode les 101. Encoder
   uniquement les nouveaux/modifiés réduirait ça à ~1 min par livre. *Effort : élevé
   (gestion des ids, suppression, fusion des .npy) — la bascule atomique actuelle rend
   déjà la ré-ingestion indolore, donc urgence faible tant que la bibliothèque bouge peu.*

## 🥉 Confort, produit, dette technique

8. **API FastAPI** au-dessus de `rechercher()`/`repondre()` (déjà en feuille de route) —
   pour intégrer BookMind dans une autre app, un site, un bot. *Effort : moyen.*
9. **Multi-bibliothèques** (une collection = un dossier) — déjà en feuille de route.
10. **Export de conversation** en Markdown/PDF depuis le chat. *Effort : faible.*
11. **Tests automatisés** (pytest) sur le chunker, la fusion RRF, les gardes d'index —
    le projet n'a aucun test ; les régressions se détectent à la main. *Effort : moyen.*
12. **Paralléliser la conversion PDF** (multiprocessing) — l'ingestion y passe plusieurs
    minutes en série. *Effort : faible-moyen.*
13. **Compréhension des figures/schémas** via LLM vision local (déjà en feuille de route,
    réaliste maintenant : qwen2.5vl:3b ou llava tournent sur 16 Go). *Effort : élevé.*

## 🔧 Environnement (hors code)

- **Le blocage réseau** (HF, pypi, OCSP GitHub) limite plusieurs axes ci-dessus et a déjà
  coûté du temps — identifier le pare-feu/antivirus responsable serait rentable.
- **VS Code** : l'interpréteur du venv est maintenant configuré (`.vscode/settings.json`) —
  recharger la fenêtre VS Code pour que les fausses erreurs « numpy could not be
  resolved » disparaissent.
