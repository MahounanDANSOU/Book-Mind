#!/usr/bin/env python3
"""Appel au LLM local (Ollama) pour générer une réponse à partir des chunks retrouvés."""
import re

import requests

from config import CONFIG


class OllamaError(RuntimeError):
    """Échec d'un appel au LLM local (Ollama absent, timeout, RAM insuffisante...).
    Le message est déjà rédigé pour être affiché tel quel à l'utilisateur."""


def joli_titre(titre: str) -> str:
    """Nom de fichier brut → titre lisible : 'delivrance-du-peche_interieur_2025...'
    → 'Delivrance du peche'. Utilisé à la fois pour l'affichage (app) et pour les
    étiquettes de sources données au LLM — s'il reçoit des noms de fichiers illisibles,
    le modèle les remplace de lui-même par des titres trouvés dans le texte, et les
    citations ne correspondent plus aux étiquettes fournies."""
    base = re.sub(r"_interieur.*$|_\d{8,}.*$", "", titre)
    base = base.replace("-", " ").replace("_", " ").strip()
    return base[:1].upper() + base[1:] if base else titre

PROMPT_TEMPLATE = """Tu es un assistant qui répond en t'appuyant UNIQUEMENT sur les \
extraits de livres fournis ci-dessous. Tu es utile et concret : tu exploites tout ce que les \
extraits contiennent de pertinent plutôt que de refuser.
Réponds dans la LANGUE DE LA QUESTION (français, anglais, espagnol...), même si les \
extraits sont dans une autre langue — traduis alors ce que tu en tires.

RÈGLES :
1. Utilise seulement le contenu des extraits — aucune connaissance extérieure, aucune invention.
2. Réponds directement à la question, puis développe avec les détails utiles des extraits.
3. Cite tes sources entre crochets après tes affirmations, en recopiant EXACTEMENT \
l'étiquette [titre, page N] donnée en tête de l'extrait utilisé — n'invente jamais \
d'autre libellé de source.
4. Si la question demande QUELS livres traitent d'un sujet, nomme les livres d'où viennent les \
extraits pertinents et résume en une phrase ce que chacun en dit.
5. Si les extraits n'abordent le sujet que partiellement ou indirectement, réponds quand même \
avec ce qu'ils contiennent de plus proche, en le précisant — ne refuse pas tant qu'il y a un lien.
6. Seulement si AUCUN extrait n'a le moindre rapport avec la question, écris : \
« Je n'ai rien trouvé sur ce sujet précis dans ta bibliothèque. » puis indique brièvement de quoi \
parlent malgré tout les extraits.

{historique}EXTRAITS :
{contexte}

QUESTION : {question}

RÉPONSE (dans la langue de la question, sourcée) :"""


ANALYSE_TEMPLATE = """Tu analyses un extrait de document pour un lecteur qui a posé une question.

EXTRAIT (source : {titre}) :
{texte}

QUESTION : {question}

Décris en 2 ou 3 phrases ce que cet extrait contient en lien avec le THÈME de la question, \
même s'il ne répond pas complètement à la question elle-même. \
Commence directement par le contenu trouvé, sans phrase d'introduction. \
Cite un ou deux courts passages de l'extrait entre guillemets « » pour appuyer ton analyse. \
Uniquement si l'extrait n'a aucun rapport, avec le thème de la question, \
réponds exactement : « Cet extrait ne parle pas directement de ce sujet. »

ANALYSE :"""


# Familles de modèles qui acceptent le paramètre Ollama "think" (raisonnement
# interne avant la réponse — améliore nettement la synthèse, au prix de latence).
_MODELES_RAISONNEURS = ("qwen3", "deepseek-r1", "magistral", "gpt-oss")

# Tokens supplémentaires accordés quand le modèle réfléchit : la réflexion compte
# dans num_predict, sans cette marge elle mangerait le budget de la réponse.
_MARGE_REFLEXION = 1200


def _raisonnement_actif() -> bool:
    """Le raisonnement est-il activé (config llm_raisonnement) ET supporté par le modèle ?"""
    return (bool(CONFIG.get("llm_raisonnement", True))
            and any(m in CONFIG["llm_model"] for m in _MODELES_RAISONNEURS))


def _appel_ollama(prompt: str, num_predict: int, num_ctx: int | None = None,
                  timeout: int = 480, temperature: float | None = None) -> str:
    """Appelle le LLM local et retourne sa réponse. Lève OllamaError (message déjà
    lisible par l'utilisateur) en cas de panne — jamais de message d'erreur retourné
    comme s'il s'agissait d'une réponse, pour que l'app puisse distinguer les deux
    (et ne pas mettre en cache une erreur transitoire).

    Température basse par défaut (config llm_temperature) : en RAG, on veut des
    réponses factuelles et reproductibles, pas de la créativité.
    Le mode réflexion suit la config (llm_raisonnement) pour TOUS les appels —
    la réflexion n'apparaît pas dans la réponse, seul le résultat final est retourné."""
    if num_ctx is None:
        num_ctx = int(CONFIG.get("llm_num_ctx", 8192))
    if temperature is None:
        temperature = float(CONFIG.get("llm_temperature", 0.2))
    raisonner = _raisonnement_actif()
    if raisonner:
        num_predict += _MARGE_REFLEXION
    corps = {"model": CONFIG["llm_model"], "prompt": prompt, "stream": False,
             # keep_alive : garde le modèle en mémoire 30 min — évite de payer
             # le rechargement (~30 s+) à chaque question espacée
             "keep_alive": "30m",
             "options": {"num_predict": num_predict, "num_ctx": num_ctx,
                         "temperature": temperature, "top_p": 0.9,
                         "repeat_penalty": 1.05}}
    if any(m in CONFIG["llm_model"] for m in _MODELES_RAISONNEURS):
        corps["think"] = raisonner
    try:
        r = requests.post(
            CONFIG["ollama_url"],
            json=corps,
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        if "response" not in data:
            raise OllamaError("Réponse Ollama inattendue (champ 'response' absent) : "
                              + str(data)[:200])
        return data["response"].strip()
    except requests.exceptions.ConnectionError:
        raise OllamaError(
            "Impossible de contacter Ollama sur " + CONFIG["ollama_url"]
            + " — vérifie qu'Ollama est lancé (icône dans la barre des tâches).") from None
    except requests.exceptions.Timeout:
        raise OllamaError("Le modèle a dépassé le délai maximal. Réessaie ; si ça se "
                          "reproduit souvent, mets llm_raisonnement: false dans "
                          "config.yaml (réponses plus rapides, moins approfondies).") from None
    except requests.exceptions.HTTPError:
        try:
            detail = r.json().get("error", r.text)
        except ValueError:
            detail = r.text
        if "allocate" in detail or "memory" in detail.lower():
            raise OllamaError(
                "Ollama n'a pas assez de mémoire (RAM) pour charger le modèle. "
                "Ferme des applications gourmandes (navigateur, etc.) puis réessaie.\n\n"
                f"Détail technique : {detail}") from None
        raise OllamaError(f"Erreur Ollama ({r.status_code}) : {detail}") from None
    except requests.exceptions.RequestException as e:
        raise OllamaError(f"Erreur de communication avec Ollama : {e}") from None


_CAR_PAR_TOKEN = 3  # approximation prudente pour du français (qwen ≈ 3-3.5 car/token)


def _budget_car_contexte(reserve_tokens: int) -> int:
    """Caractères de contexte qui tiennent dans la fenêtre du modèle une fois
    réservés les tokens du gabarit de prompt et de la réponse à générer. Toutes les
    limites de taille du module dérivent d'ici : elles suivent llm_num_ctx du config."""
    num_ctx = int(CONFIG.get("llm_num_ctx", 8192))
    return max((num_ctx - reserve_tokens) * _CAR_PAR_TOKEN, 2000)


def repondre(question: str, chunks: list[dict], timeout: int = 1200,
             historique: list[tuple[str, str]] | None = None) -> str:
    """Réponse rédigée et sourcée à partir des chunks retrouvés.

    historique : échanges précédents de la conversation [(role, texte), ...] —
    permet au LLM de comprendre les questions de suivi (« et comment le pratiquer ? »)."""
    if not chunks:
        return "Aucun passage pertinent trouvé dans la base documentaire."

    # Réserve de tokens : réflexion éventuelle + réponse + gabarit + historique.
    reserve = 2600 if _raisonnement_actif() else 1200
    budget_car = _budget_car_contexte(reserve)

    bloc_hist = ""
    if historique:
        lignes = []
        for role, texte in historique[-4:]:
            prefixe = "Utilisateur" if role == "user" else "Assistant"
            lignes.append(f"{prefixe} : {texte[:400]}")
        bloc_hist = ("CONVERSATION PRÉCÉDENTE (contexte pour comprendre la question) :\n"
                     + "\n".join(lignes) + "\n\n")

    # Garde-fou : le contexte ne doit jamais déborder num_ctx, sinon Ollama tronque
    # silencieusement le DÉBUT du prompt — c'est-à-dire les instructions — et le
    # modèle se met à répondre hors sujet.
    morceaux = []
    taille = len(bloc_hist)
    for c in chunks:
        entete = (f"[{joli_titre(c['titre'])}"
                  + (f", page {c['page']}" if c.get("page") else "") + "]")
        bloc = f"{entete}\n{c['texte']}"
        if taille + len(bloc) > budget_car and morceaux:
            break
        morceaux.append(bloc)
        taille += len(bloc)
    contexte = "\n\n".join(morceaux)

    prompt = PROMPT_TEMPLATE.format(historique=bloc_hist, contexte=contexte,
                                    question=question)
    return _appel_ollama(prompt, num_predict=700, timeout=timeout)


def analyser_extrait(question: str, chunk: dict, timeout: int = 360) -> str:
    """Explique ce qu'un extrait précis dit en rapport avec la question."""
    prompt = ANALYSE_TEMPLATE.format(
        titre=joli_titre(chunk["titre"]),
        texte=chunk["texte"], question=question,
    )
    return _appel_ollama(prompt, num_predict=250, timeout=timeout, temperature=0.1)


POINTS_MAP_TEMPLATE = """Tu lis un passage d'un livre (le texte peut contenir des caractères mal encodés, \
ignore-les et concentre-toi sur le sens).

PASSAGE :
{texte}

Extrais les idées et affirmations importantes de ce passage {precision}, une par ligne, \
sans numérotation, en reformulant clairement. Le passage fait partie d'un livre : il contient \
forcément des idées à retenir, extrais-en au moins trois.

IDÉES IMPORTANTES :"""

POINTS_SYNTHESE_TEMPLATE = """Voici une liste brute d'idées extraites d'un chapitre de livre{precision}. \
Chaque bloc d'idées est précédé de la plage de pages d'où il provient, entre crochets, \
par exemple [pages 13-16].

IDÉES BRUTES :
{idees}

À partir de ces idées, rédige la liste des POINTS CLÉS ESSENTIELS à avoir compris \
avant de passer à la suite — les fondations du chapitre. \
Regroupe les redondances, garde entre 4 et 8 points maximum, classe-les par ordre d'importance.

Pour CHAQUE point, respecte STRICTEMENT ce format (une ligne par champ, dans cet ordre) :
TITRE: <une phrase courte et claire résumant le point>
EXPLICATION: <développe en 3 à 5 phrases : pourquoi ce point est important, ce qu'il signifie \
concrètement, comment il se rattache au reste du chapitre>
PAGES: <la ou les plages de pages (reprises telles quelles entre crochets dans les idées brutes) \
dont ce point est tiré>

Sépare chaque point par une ligne vide. Numérote les points (1., 2., 3. ...) avant TITRE. \
Ne rajoute rien d'autre après le dernier point.

POINTS CLÉS DU CHAPITRE :"""


def _regrouper_par_budget(chunks: list[dict], budget_car: int | None = None) -> list[dict]:
    """Regroupe les segments en blocs ne dépassant pas budget_car caractères (pour tenir dans
    la fenêtre de contexte du modèle), en gardant la plage de pages couverte par chaque bloc.
    Par défaut : dérivé de llm_num_ctx, plafonné à 12000 car. (des blocs trop gros diluent
    l'extraction d'idées — 300 tokens de sortie pour un bloc énorme)."""
    if budget_car is None:
        reserve = 2600 if _raisonnement_actif() else 800
        budget_car = min(_budget_car_contexte(reserve), 12000)
    blocs, courant, taille = [], [], 0
    for c in chunks:
        t = c["texte"]
        if taille + len(t) > budget_car and courant:
            blocs.append(courant)
            courant, taille = [], 0
        courant.append(c)
        taille += len(t)
    if courant:
        blocs.append(courant)
    resultat = []
    for bloc in blocs:
        pages = [c["page"] for c in bloc if c.get("page")]
        resultat.append({
            "texte": "\n\n".join(c["texte"] for c in bloc),
            "page_min": min(pages) if pages else None,
            "page_max": max(pages) if pages else None,
        })
    return resultat


_POINT_RE = re.compile(
    r"TITRE\s*:\s*(?P<titre>.+?)\s*\n"
    r"EXPLICATION\s*:\s*(?P<explication>.+?)\s*\n"
    r"PAGES\s*:\s*(?P<pages>.+?)\s*(?:\n|$)",
    re.IGNORECASE | re.DOTALL,
)


def _parser_points(texte_brut: str) -> list[dict]:
    """Extrait les points structurés (titre/explication/pages) de la sortie du LLM.
    Retourne [] si le format n'a pas été respecté (l'app affiche alors le texte brut)."""
    points = []
    for m in _POINT_RE.finditer(texte_brut):
        points.append({
            "titre": " ".join(m.group("titre").split()),
            "explication": " ".join(m.group("explication").split()),
            "pages": m.group("pages").strip(),
        })
    return points


def points_cles_chapitre(chunks: list[dict], sujet: str = "", timeout: int = 600):
    """Extrait les points clés essentiels d'un chapitre (ensemble de chunks d'une plage de pages).
    Procède en map-reduce pour absorber un chapitre plus long que la fenêtre de contexte :
    1) extrait les idées de chaque bloc de texte (en conservant la plage de pages d'origine),
    2) synthétise en points clés développés, chacun rattaché à sa plage de pages source.

    Retourne soit une liste de dicts {titre, explication, pages} (cas normal), soit une chaîne
    (texte brut si le LLM n'a pas respecté le format demandé, ou message si rien à traiter).
    Lève OllamaError si un appel au LLM échoue (panne, timeout, RAM)."""
    if not chunks:
        return "Aucun texte trouvé pour ce livre et cette plage de pages."

    precision = f"en lien avec le sujet « {sujet} »" if sujet.strip() else ""
    precision_synth = f" (sujet ciblé : « {sujet} »)" if sujet.strip() else ""

    blocs = _regrouper_par_budget(chunks)
    idees = []
    for bloc in blocs:
        prompt = POINTS_MAP_TEMPLATE.format(texte=bloc["texte"], precision=precision)
        res = _appel_ollama(prompt, num_predict=300, timeout=timeout)
        if res and res.strip().lower() not in ("(rien)", "rien"):
            if bloc["page_min"] is not None:
                idees.append(f"[pages {bloc['page_min']}-{bloc['page_max']}]\n{res}")
            else:
                idees.append(res)

    if not idees:
        return "Le chapitre n'a pas produit d'idées exploitables (texte trop court ou vide)."

    idees_texte = "\n\n".join(idees)
    # Garde-fou : si la liste brute est énorme, on la tronque pour la synthèse finale
    idees_texte = idees_texte[:_budget_car_contexte(2700 if _raisonnement_actif() else 1500)]
    prompt = POINTS_SYNTHESE_TEMPLATE.format(idees=idees_texte, precision=precision_synth)
    brut = _appel_ollama(prompt, num_predict=900, timeout=timeout)
    points = _parser_points(brut)
    return points if points else brut


if __name__ == "__main__":
    import sys
    from search import rechercher, IndexAbsentError
    q = sys.argv[1] if len(sys.argv) > 1 else "test"
    try:
        print(repondre(q, rechercher(q)))
    except (OllamaError, IndexAbsentError) as e:
        print(f"ERREUR : {e}")
        sys.exit(1)