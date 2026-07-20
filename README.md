# BookMind

Assistant documentaire local : posez une question en langage naturel sur une bibliothèque de documents (PDF, `.txt`, `.md`) et obtenez une réponse rédigée et sourcée, ou une liste des documents les plus pertinents.

Conçu pour un principe simple : **aucune donnée ne quitte votre machine**. Pas d'appel à une API IA tierce (OpenAI, Google, etc.) — la recherche et la génération tournent entièrement en local. Ce dépôt peut servir de base pour construire votre propre outil du même genre.

## Comment ça marche

```text
Documents (PDF/.txt/.md)
        │
        ▼
   ingest.py ──► découpe en chunks ──► SQLite + FTS5 (index lexical)
                                   └──► embeddings (sentence-transformers) → .npy
        │
        ▼
   search.py ──► recherche hybride (BM25 + similarité cosinus, fusion RRF)
        │
        ▼
  generate.py ──► LLM local (Ollama) ──► réponse rédigée, sourcée
        │
        ▼
    app.py ──► interface Streamlit
```

- **Recherche lexicale** (FTS5/BM25) : bonne sur les termes exacts, sigles, formules.
- **Recherche sémantique** (embeddings `bge-m3` servis par Ollama, 1024 dimensions) : bonne sur le sens, les reformulations.
- **Fusion RRF (Reciprocal Rank Fusion)** : combine les deux classements sans avoir à pondérer arbitrairement.
- **Génération** : le LLM (Ollama, `qwen3:8b`, avec mode raisonnement) ne répond qu'à partir des extraits retrouvés, avec citation des sources — pas de réponse inventée hors contexte.

## Fonctionnalités de la plateforme

Tour d'horizon de tout ce que fait BookMind aujourd'hui, expliqué avec les termes techniques (utile pour un rapport de stage ou pour comprendre le code).

### 1. Ingestion documentaire (`ingest.py`)

C'est l'étape qui transforme des fichiers bruts en une base interrogeable.

- **Formats supportés** : `.pdf`, `.txt`, `.md`. Les PDF sont convertis en texte via **PyMuPDF** (`fitz`), avec repérage du **numéro de page d'origine** pour chaque passage — utile pour citer une source précisément ou cibler un chapitre.
- **Chunking (découpage en segments)** : chaque document est découpé en blocs d'environ 350 mots (`chunk_size_mots`, réglable dans `config.yaml`), avec un chevauchement de 60 mots (`chunk_overlap_mots`) entre segments consécutifs — ce chevauchement évite qu'une idée à cheval sur deux segments soit coupée et perdue pour la recherche.
- **Double indexation** : chaque segment est stocké deux fois, sous deux formes complémentaires :
  - dans une table **SQLite + FTS5** (Full-Text Search 5), pour la recherche par mots-clés exacts ;
  - sous forme d'**embedding** (vecteur numérique produit par le modèle `bge-m3` via Ollama), sauvegardé dans un fichier `.npy`, pour la recherche par similarité de sens.
- **Robustesse** : les erreurs MuPDF non fatales (ex. annotations multimédia mal formées) sont silencieuses ; un PDF ou une page défectueuse n'interrompt pas le traitement des autres documents.
- **OCR en secours** (optionnel, voir section dédiée) : si une page n'a pas de couche texte (PDF scanné, texte piégé dans une image), elle est automatiquement passée à **PaddleOCR** (moteur local) pour en extraire le texte.
- **Détection des changements** : un manifeste (`data/db/manifest.json`) retient l'état de `data/raw/` (taille + date de chaque fichier). `python ingest.py --si-nouveau` ne relance l'ingestion que si quelque chose a changé — utilisé automatiquement par `start_app.bat` et par le bouton de rechargement de l'app.
- **Fichiers annexes ignorés** : les fiches `.metadata` (métadonnées de provenance d'un livre, sans contenu réel) sont exclues de l'ingestion pour ne pas polluer les résultats de recherche.
- **Détection automatique du sommaire (chapitres)** : pour chaque PDF, `extraire_sommaire()` tente deux méthodes, dans l'ordre :
  1. les **signets natifs du PDF** (la table des matières intégrée, quand l'éditeur du document l'a incluse), en écartant les faux sommaires du type "un signet par diapositive" (PDF issus de PowerPoint) ;
  2. à défaut, une **détection typographique** : analyse de la taille de police de chaque ligne du document pour repérer la taille utilisée pour les titres de chapitre, puis reconnaissance du motif "Chapitre N" à cette taille.

  Le résultat (titre de chapitre + page de début) est stocké dans une table `chapitres`, et sert à peupler un sélecteur de chapitres par leur titre dans l'app — sans que l'utilisateur ait besoin de connaître les numéros de page. Si aucune des deux méthodes n'aboutit, l'app propose en dernier recours une plage de pages manuelle.

### 2. Recherche hybride (`search.py`)

La brique qui retrouve les passages pertinents pour une question, sans jamais appeler le LLM (rapide, quasi instantané).

- **Recherche lexicale (BM25)** : interroge l'index FTS5 avec les mots de la question — excellente sur les termes exacts, sigles, formules, noms propres.
- **Recherche sémantique (cosinus)** : compare l'embedding de la question à ceux de tous les segments — retrouve un passage même si la question est reformulée avec d'autres mots que le texte source.
- **Fusion RRF (Reciprocal Rank Fusion)** : combine les deux classements en un seul score, sans avoir à choisir arbitrairement un poids entre lexical et sémantique — chaque segment gagne des points selon son *rang* dans chaque liste, pas selon son score brut (qui n'est pas comparable entre BM25 et cosinus).
- **Deux fonctions de recherche** :
  - `rechercher()` : renvoie les meilleurs segments (top-k, réglable), pour construire une réponse.
  - `rechercher_documents()` : agrège les scores par livre entier, pour répondre à "quels documents parlent de X ?".
- **Fonctions d'accès pour le mode chapitre** :
  - `lister_livres()` : liste les livres correctement paginés avec leur plage de pages et leur nombre de chapitres détectés, pour peupler un sélecteur.
  - `lister_chapitres()` : sommaire détecté d'un livre (titre + plage de pages calculée automatiquement pour chaque chapitre) — permet à l'app d'offrir un sélecteur de chapitres par titre plutôt que de faire saisir des numéros de page.
  - `chunks_par_plage()` : récupère, dans l'ordre du livre, tous les segments d'une plage de pages donnée — sert à traiter un chapitre entier plutôt qu'un simple top-k.

### 3. Génération par LLM local (`generate.py`)

Tous les appels passent par **Ollama**, un serveur qui fait tourner un modèle de langage (`qwen3:8b`) entièrement sur la machine — aucune donnée n'est envoyée à un tiers. Pour la réponse principale, le **mode raisonnement** de Qwen3 est activé (`llm_raisonnement` dans config.yaml) : le modèle réfléchit avant de rédiger — meilleure synthèse, quelques dizaines de secondes de plus. Les tâches en masse (analyse par source, points clés) le désactivent pour rester rapides.

- **Réponse sourcée (`repondre`)** : le LLM ne reçoit que les segments retrouvés par la recherche hybride et pour instruction explicite de ne répondre qu'à partir de ces extraits, en citant sa source entre crochets `[titre, section]`. S'il ne trouve pas l'information, il doit le dire plutôt qu'inventer.
- **Analyse par source (`analyser_extrait`)** : sur demande (case à cocher dans l'app), chaque source utilisée est en plus analysée individuellement — le LLM explique ce que *cet extrait précis* apporte par rapport à la question, avec citations entre guillemets. Utile quand la réponse globale est courte mais qu'on veut comprendre le détail de chaque source.
- **Points clés de chapitre (`points_cles_chapitre`)** : traite un **chapitre entier** (pas un top-k) via une stratégie **map-reduce** — le chapitre est d'abord découpé en lots qui tiennent dans la fenêtre de contexte du modèle (en conservant la plage de pages de chaque lot), chaque lot est résumé en idées brutes ("map"), puis toutes ces idées sont fusionnées en 4 à 8 points clés hiérarchisés ("reduce"). Chaque point clé est **développé** (3 à 5 phrases : pourquoi il compte, ce qu'il signifie concrètement) et rattaché aux **pages dont il est tiré**. L'app peut alors afficher, pour chaque point, le ou les passages associés — le texte affiché est toujours celui réellement stocké en base (retrouvé par numéro de page), jamais un extrait généré par le LLM, pour éliminer tout risque de citation inventée.
- **Gestion d'erreurs robuste** : les pannes réseau, timeouts et manques de mémoire RAM d'Ollama sont interceptés et traduits en messages clairs (pas de traceback brut affiché à l'utilisateur).

### 4. Interface utilisateur (`app.py`, Streamlit)

**Une seule interface : la conversation** (comme ChatGPT, mais basée uniquement sur ta bibliothèque). Chaque question déclenche une recherche sur tout le corpus puis une **réponse raisonnée, organisée et sourcée** générée par le LLM local (1 à 3 minutes sur ce PC). Sous chaque réponse : la liste des **livres consultés** et les **passages exacts** dépliables. L'historique est conservé (les questions de suivi sont comprises) ; bouton « 🗑️ Nouvelle conversation » pour repartir de zéro.

Dans la barre latérale, l'outil **📖 Points clés d'un chapitre** : choisis un livre puis un **chapitre par son titre** (sommaire détecté automatiquement à l'ingestion), + sujet optionnel — le résultat (points clés développés, rattachés à leurs pages, passages dépliables) s'affiche **dans la conversation**, comme n'importe quelle réponse.

Le LLM reçoit pour consigne de **n'utiliser que les extraits fournis** et de citer ses sources `[titre, page]`. S'il ne trouve rien de pertinent, il le dit — il n'invente jamais de contenu hors bibliothèque. Le **mode raisonnement** (`llm_raisonnement` dans config.yaml) s'applique à **tous** les appels : réponses, analyses, points clés.
- **Rechargement à chaud** : bouton dans la barre latérale qui détecte les nouveaux documents, relance l'ingestion si besoin, et recharge l'index — sans redémarrer le serveur.
- **Mise en cache** (`st.cache_data`) : la recherche, la génération de réponse et les analyses par source sont mises en cache, pour éviter de relancer des calculs coûteux à chaque interaction de l'interface (Streamlit réexécute tout le script à chaque clic).

### 5. Lancement automatisé (`start_app.bat`, `setup.bat`, `setup_ocr.bat`)

- **`setup.bat`** : installation complète en une commande — environnement virtuel, dépendances, dossiers de données, téléchargement des deux modèles (embeddings + LLM), **et installation automatique de PaddleOCR**.
- **`setup_ocr.bat`** : installation de l'OCR (**PaddleOCR** + son moteur `paddlepaddle`). Appelée automatiquement par `setup.bat`, ou à relancer seul si réinstallation nécessaire.
- **`start_app.bat`** : à chaque lancement, détecte et ingère les nouveaux documents, vérifie qu'Ollama tourne (le démarre automatiquement sinon), puis lance le serveur Streamlit.

### 6. Évaluation objective (`evaluate.py`)

Mesure la qualité de la recherche sur un jeu de questions/réponses réel (`data/eval.jsonl`) :

- **Recall@5** : le bon document figure-t-il dans les 5 premiers résultats ?
- **MRR (Mean Reciprocal Rank)** : à quel rang moyen le bon document apparaît-il ?

Permet de régler objectivement `chunk_size_mots`, `top_k`, etc. plutôt qu'à l'aveugle.

### 7. Confidentialité par conception

Principe transversal à toute la plateforme : aucune donnée ne quitte la machine (pas d'API IA tierce), et aucun document source ni sa transformation (chunks, embeddings, base SQLite) n'est jamais commité ou publié — appliqué via `.gitignore` et vérifié systématiquement.

## Installation de A à Z

Guide complet pour reproduire le projet sur une machine Windows vierge. Comptez ~15 minutes plus le temps de téléchargement des modèles (~4,8 Go au total).

### Étape 0 — Prérequis à installer

| Outil | Où le trouver | Point d'attention |
|---|---|---|
| **Python 3.11+** | <https://www.python.org/downloads/> | Cochez **"Add python.exe to PATH"** pendant l'installation |
| **Git** | <https://git-scm.com/downloads> | Options par défaut |
| **Ollama** | <https://ollama.com/download> | Lancez-le une fois après installation (icône dans la barre des tâches) |

Vérifiez dans un terminal `cmd` que tout est accessible :

```cmd
python --version
git --version
ollama --version
```

Chaque commande doit afficher un numéro de version. Sinon, réinstallez l'outil concerné en vérifiant l'option PATH.

### Étape 1 — Récupérer le projet

```cmd
cd %USERPROFILE%
git clone https://github.com/MahounanDANSOU/Book-Mind
cd Book-Mind
```

(ou n'importe quel autre dossier que `%USERPROFILE%` — tout le reste se fait depuis le dossier du projet)

### Étape 2 — Installation automatique (recommandé)

```cmd
setup.bat
```

Ce script fait tout, dans l'ordre, avec messages d'erreur explicites :

1. crée l'environnement virtuel Python (`venv\`) ;
2. installe les dépendances de `requirements.txt` (sentence-transformers, streamlit, pymupdf, etc.) ;
3. crée les dossiers de données (`data\raw\`, `data\db\`) et le fichier `data\eval.jsonl` ;
4. télécharge le modèle d'embeddings `bge-m3` (~1,2 Go) via Ollama ;
5. télécharge le LLM `qwen3:8b` (~5,2 Go) via Ollama.

### Étape 2 bis — Installation manuelle (équivalent commande par commande)

Si vous préférez comprendre/contrôler chaque étape, voici exactement ce que fait `setup.bat` :

```cmd
:: 1. Environnement virtuel
python -m venv venv
venv\Scripts\activate

:: 2. Dépendances Python
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 3. Dossiers et fichiers de données
mkdir data\raw
mkdir data\db
type nul > data\eval.jsonl

:: 4. Modèle d'embeddings (téléchargement unique — nécessite internet)
ollama pull bge-m3

:: 5. Modèle LLM (téléchargement unique — nécessite internet)
ollama pull qwen3:8b
```

Après ça, tout fonctionne **hors ligne** : les deux modèles sont en cache local et le code n'accède plus jamais à internet.

### Étape 3 — Ajouter vos documents

Copiez vos fichiers `.pdf`, `.txt` ou `.md` dans `data\raw\` (explorateur Windows ou terminal) :

```cmd
copy "C:\chemin\vers\mon-livre.pdf" data\raw\
```

### Étape 4 — Lancer l'application

```cmd
start_app.bat
```

Ce script enchaîne automatiquement :

1. activation du venv ;
2. **détection des documents nouveaux/modifiés dans `data\raw\`** et ingestion si besoin (découpage en chunks, conversion PDF → `.md`, calcul des embeddings — plusieurs minutes la première fois) ;
3. vérification qu'Ollama tourne, **démarrage automatique sinon** ;
4. lancement du serveur Streamlit — l'onglet s'ouvre tout seul dans le navigateur sur `http://localhost:8501`.

Lancement manuel équivalent (sans les automatismes) :

```cmd
venv\Scripts\activate
python ingest.py --si-nouveau
streamlit run app.py
```

(et non `streamlit app.py` ni `python app.py` — la sous-commande `run` est obligatoire)

## Utilisation

Trois modes dans l'interface :

- **Question précise** : réponse rédigée par le LLM local, avec sources. Peut prendre 1 à 3 minutes sur un PC sans carte graphique dédiée — c'est le prix de la génération 100 % locale. Une case à cocher **"🔍 Détailler ce que chaque source dit sur la question"** permet en plus de faire analyser chaque extrait un par un par le LLM (~30 s à 1 min par source).
- **Quels documents parlent de...?** : liste des documents pertinents, sans passer par le LLM — quasi instantané.
- **Points clés d'un chapitre** : choisissez un livre et une **plage de pages** (le chapitre, lu sur la table des matières), éventuellement un sujet à cibler, et l'app extrait les **points clés essentiels** — les fondations à maîtriser avant d'avancer. Contrairement aux deux autres modes qui piochent les meilleurs extraits, celui-ci lit **tout** le chapitre (traitement par lots puis synthèse), donc comptez plusieurs minutes pour un long chapitre. Le repérage se fait par page car les livres n'ont pas de marqueur de chapitre exploitable ; les numéros de page, eux, sont fiables.

### Ajouter des documents sans redémarrer le serveur

1. Copiez vos nouveaux fichiers dans `data\raw\`.
2. Cliquez sur **"🔄 Recharger l'index documentaire"** dans la barre latérale.

Le bouton détecte les changements dans `data\raw\` (via `data\db\manifest.json`) : s'il y a du nouveau, il relance l'ingestion complète puis recharge l'index à la volée ; sinon il recharge simplement l'index. Pas besoin de couper/relancer le serveur.

### Ingestion à la main

```cmd
python ingest.py                # ingestion complète inconditionnelle
python ingest.py --si-nouveau   # ingère seulement si data\raw\ a changé
```

Chaque PDF converti génère un fichier `_converti_<nom>.md` dans `data\raw\`, utile pour vérifier visuellement la qualité de l'extraction (notamment pour les PDF issus de PowerPoint).

**⚠️ L'ingestion réingère tout depuis zéro à chaque fois** (pas d'ajout incrémental) — la détection évite seulement de la relancer inutilement quand rien n'a changé.

L'index est reconstruit dans des fichiers temporaires puis **basculé atomiquement** à la toute fin : pendant toute la ré-ingestion, l'app continue de répondre sur l'ancien index, complet et cohérent. Un échec en cours de route laisse l'ancien index intact.

## Texte dans les images, PDF scannés (OCR) et diagrammes

Par défaut, l'ingestion lit la **couche texte** des PDF (rapide, fidèle). Les erreurs MuPDF non fatales (ex. `cannot create appearance stream for Screen annotations`, dues à des annotations multimédia dans le PDF) sont silencieuses et sans conséquence : aucune page n'est perdue, et un PDF ou une page défectueuse n'interrompt plus l'ingestion des autres.

Trois niveaux de récupération du contenu, du plus simple au plus lourd :

1. **Texte natif** (actif) — fonctionne pour tout PDF ayant une couche texte. C'est le cas de la quasi-totalité des PDF générés depuis un traitement de texte ou un export PowerPoint.
2. **OCR** (texte piégé dans des images, PDF scannés) — **inclus et activable** sur ce projet. Utilise **PaddleOCR** (moteur OCR local, 100 % hors ligne) :
   - Bonne précision sur texte dégradé ou mal aligné, en français comme en anglais
   - Version CPU par défaut ; sur GPU Nvidia, installez `paddlepaddle-gpu` à la place de `paddlepaddle` pour accélérer nettement
   - Installation simple : `pip install paddleocr paddlepaddle` (fait automatiquement par `setup.bat` via `setup_ocr.bat`)

   Contrôlé par `ocr_fallback: true` dans `config.yaml`. Pendant l'ingestion, chaque page sans texte natif est automatiquement passée à l'OCR (message `OCR appliqué à N page(s)` affiché). Le modèle PaddleOCR français est téléchargé et mis en cache à la première utilisation (~100 Mo).

3. **Compréhension des diagrammes/schémas** (décrire ce qu'une figure *signifie*, pas juste le texte qu'elle contient) — nécessite un **LLM multimodal (vision)**. Trop lourd pour 8 Go de RAM : à réserver à une machine avec GPU ou au futur VPS. Principe : lors de l'ingestion, chaque figure serait envoyée à un modèle vision local (ex. `llava` ou `qwen2.5vl` via Ollama) qui en produit une description textuelle, ensuite indexée comme le reste. Voir la feuille de route — non implémenté à ce jour.

## Configuration

Tout se règle dans `config.yaml` :

```yaml
embedding_backend: ollama  # 'ollama' (recommandé) ou 'sentence-transformers' (modèle HF local)
embedding_model: bge-m3
llm_model: qwen3:8b
llm_raisonnement: true     # le modèle réfléchit avant de répondre (qwen3 & co)
chunk_size_mots: 250       # taille d'un chunk, en mots (phrases entières regroupées)
chunk_overlap_mots: 50     # chevauchement entre chunks consécutifs
top_k: 8                   # nombre de chunks utilisés pour générer une réponse
llm_temperature: 0.2       # température LLM (basse = réponses factuelles et stables)
llm_num_ctx: 8192          # fenêtre de contexte du LLM
db_path: data/db/logos.db
emb_path: data/db/embeddings.npy
ids_path: data/db/chunk_ids.npy
raw_dir: data/raw
eval_path: data/eval.jsonl
ollama_url: http://localhost:11434/api/generate
ocr_fallback: true         # OCR de secours pour les pages sans couche texte
ocr_langue: fr             # code langue PaddleOCR
```

Les chemins sont relatifs à la racine du projet (résolus par `config.py`, quel que soit le dossier d'où le script est lancé).

Si vous changez `llm_model`, récupérez-le d'abord (`ollama pull <modèle>`). Si vous changez `embedding_model` ou `embedding_backend`, il faut en plus **ré-ingérer** (`python ingest.py`) : l'index mémorise le modèle qui l'a construit (`data/db/index_meta.json`) et refuse de servir avec un modèle différent — c'est ce qui évite des résultats aberrants silencieux.

## Évaluer la qualité de la recherche

Avant de considérer le système comme fiable, validez-le objectivement :

1. Remplissez `data/eval.jsonl` avec de vraies questions/réponses tirées de vos documents (un objet JSON par ligne). Pour démarrer sans effort, générez automatiquement des questions depuis vos livres avec le LLM local :

   ```cmd
   python scripts\generer_eval.py 15
   ```

   (échantillonne 15 livres, tire un passage au milieu de chacun, et fait écrire au LLM une question dont la réponse est dans ce passage — le document attendu est donc connu par construction). Format d'une ligne :

   ```json
   {"question": "...", "doc_attendu": "nom_du_fichier_sans_extension", "mot_cle_extrait": "expression exacte tirée du texte"}
   ```

   `doc_attendu` doit correspondre exactement au nom du fichier source sans extension (c'est ce que la recherche retourne comme `titre`).

2. Lancez :

   ```cmd
   python evaluate.py
   ```

   Mesure le **Recall@5** (le bon document est-il dans les 5 premiers résultats ?) et le **MRR** (à quel rang en moyenne). Seuil recommandé avant de considérer la recherche fiable : **Recall@5 ≥ 80%**.

**Ne sautez pas cette étape** : sans jeu de questions réel, aucune mesure objective n'est possible, et le tuning (taille des chunks, top_k, etc.) devient à l'aveugle.

## Structure du projet

```text
BookMind/
├── setup.bat         # installation complète (venv, dépendances, dossiers, modèles + OCR)
├── setup_ocr.bat     # installation isolée de PaddleOCR (appelée par setup.bat)
├── start_app.bat     # lance l'app (ingestion auto si nouveaux docs, check Ollama, streamlit)
├── app.py            # interface Streamlit (3 modes)
├── config.py         # chargement centralisé de config.yaml (chemins résolus, validation)
├── ingest.py         # ingestion : PDF/txt/md → chunks → DB + embeddings (+ détection --si-nouveau, OCR PaddleOCR)
├── search.py         # recherche hybride (BM25 + sémantique + RRF) + accès par livre/plage de pages
├── generate.py       # appels au LLM local (Ollama) : réponse globale, analyse par source, points clés
├── evaluate.py       # mesure Recall@5 / MRR sur data/eval.jsonl
├── config.yaml       # configuration centrale
├── requirements.txt
├── scripts/
│   ├── pdf_to_md.py      # conversion PDF → Markdown en standalone (debug qualité)
│   └── generer_eval.py   # génère data/eval.jsonl automatiquement via le LLM local
├── data/                     ← créé par setup.bat, jamais versionné
│   ├── raw/          # vos documents sources (+ _converti_*.md générés)
│   ├── eval.jsonl    # jeu de questions/réponses pour evaluate.py
│   └── db/           # base SQLite + embeddings + manifest.json (générés)
└── .streamlit/
    └── config.toml   # config Streamlit (fileWatcherType none — relancer après modif du code)
```

**⚠️ Confidentialité** : `data/` entier est exclu de git (`.gitignore`) — vos documents et toutes leurs transformations (`_converti_*.md`, base, embeddings) ne doivent jamais être commités ni publiés.

## Dépannage

- **`python` introuvable après installation** : Python n'est pas dans le PATH — réinstallez en cochant "Add python.exe to PATH", ou utilisez `py` au lieu de `python`.
- **`streamlit app.py` → "No such command"** : il manque la sous-commande `run` → `streamlit run app.py`.
- **L'ingestion échoue avec « modèle absent » sur les embeddings** : le modèle bge-m3 n'a pas été téléchargé — lancez `ollama pull bge-m3`. (Si vous utilisez le backend `sentence-transformers`, le modèle HF doit être en cache local — téléchargez-le une fois avec `set HF_HUB_OFFLINE=0`.)
- **Erreur réseau (`getaddrinfo failed`) alors que les modèles sont déjà en cache** : bug connu de certaines versions de `huggingface_hub` qui retentent un appel réseau avant de retomber sur le cache local. Le code force déjà `local_files_only=True` pour éviter ce problème.
- **Les nouveaux documents n'apparaissent pas dans les réponses** : cliquez sur "🔄 Recharger l'index documentaire" (voir plus haut).
- **Une modification du code ne s'applique pas dans l'app** : `fileWatcherType = "none"` désactive le rechargement à chaud — arrêtez le serveur (Ctrl+C) et relancez `start_app.bat`.
- **"Impossible de contacter Ollama"** : si vous êtes passé par `streamlit run app.py` directement (sans `start_app.bat`), vérifiez qu'Ollama tourne (icône dans la barre des tâches) et que `ollama_url` dans `config.yaml` est correct.
- **"Ollama n'a pas assez de mémoire (RAM)"** : le modèle 8b a besoin d'environ 5,5 Go libres pour se charger. Fermez les applications gourmandes (navigateurs surtout) et réessayez. Si le problème persiste sur une machine à 8 Go, envisagez un modèle plus léger dans `config.yaml` (ex. `qwen3:4b`, à récupérer d'abord via `ollama pull qwen3:4b`) — un cran moins bon, mais empreinte mémoire divisée par deux et deux fois plus rapide sur CPU.

## Feuille de route

- [x] Ingestion PDF avec repérage de page
- [x] Recherche hybride lexicale + sémantique (RRF)
- [x] Génération locale sourcée (Ollama) + analyse détaillée par source
- [x] Validation objective (Recall@5/MRR)
- [x] Rechargement à chaud + détection automatique des nouveaux documents (au démarrage et via le bouton de l'app)
- [x] Installation reproductible en un script (`setup.bat`)
- [x] Extraction PDF robuste (erreurs MuPDF non fatales silencieuses, page par page) + OCR optionnel (PaddleOCR, GPU-accéléré)
- [x] Détection automatique du sommaire par livre (signets PDF natifs ou détection typographique) — sélection d'un chapitre par son titre, sans consulter le livre
- [ ] Compréhension des diagrammes/figures via LLM multimodal local (llava/qwen-vl) — nécessite GPU/VPS
- [ ] Ingestion incrémentale (à faire — actuellement réingestion complète dès qu'un changement est détecté)
- [ ] Migration vers une API (FastAPI) par-dessus les mêmes fonctions `rechercher()`/`repondre()`, Streamlit servant uniquement d'interface de test
- [ ] Multi-bibliothèques : une collection = un dossier (`data/<collection>/raw` + `db`), paramètre de collection dans l'API
- [ ] Déploiement sur serveur auto-hébergé (VPS type Hetzner/OVH/Scaleway) avec Ollama sur la même machine, HTTPS + authentification — pour une intégration dans une application tierce sans jamais faire transiter les données par un service IA externe

## Principe de conception

Le choix de tout héberger localement (modèle d'embeddings, base de données, LLM) n'est pas une contrainte technique temporaire mais un principe : **aucune donnée ne doit transiter par un service tiers qui pourrait la lire, la stocker ou s'en servir pour entraîner un modèle**. Ce principe reste valable même dans une architecture de service accessible à distance (VPS auto-hébergé, HTTPS) — ce qui compte c'est que le LLM et les données restent sous votre contrôle, pas que la machine soit physiquement déconnectée d'internet.
