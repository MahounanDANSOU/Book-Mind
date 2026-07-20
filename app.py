import re
import subprocess
import sys

import streamlit as st

from config import RACINE
from embeddings import EmbeddingError
from search import (rechercher, recharger_index, lister_livres,
                    lister_chapitres, chunks_par_plage, IndexAbsentError)
from generate import repondre, points_cles_chapitre, OllamaError, joli_titre as _joli_titre
from ingest import ingestion_necessaire


# Caches : Streamlit réexécute tout le script à chaque interaction ; sans cache,
# la moindre interaction relancerait des calculs coûteux.
@st.cache_data(show_spinner=False)
def _livres_cache():
    return lister_livres()


@st.cache_data(show_spinner=False)
def _chapitres_cache(titre: str):
    return lister_chapitres(titre)


@st.cache_data(show_spinner=False)
def _points_cles_cache(titre: str, page_debut: int, page_fin: int, sujet: str):
    chunks = chunks_par_plage(titre, page_debut, page_fin)
    return chunks, points_cles_chapitre(chunks, sujet)


def _passages_pour_pages(chunks: list, pages_str: str) -> list:
    """Retrouve, parmi les segments réellement stockés en base, ceux dont la page correspond
    aux plages mentionnées par le LLM (ex. '13-16, 23-25') — jamais de texte inventé."""
    plages = []
    for deb, fin in re.findall(r"(\d+)\s*(?:-|à|–)\s*(\d+)", pages_str):
        plages.append((int(deb), int(fin)))
    if not plages:
        for n in re.findall(r"\d+", pages_str):
            plages.append((int(n), int(n)))
    return [c for c in chunks if c.get("page") is not None
            and any(deb <= c["page"] <= fin for deb, fin in plages)]


def _afficher_sources(chunks: list) -> None:
    """Livres consultés + passages exacts (dépliables) sous une réponse."""
    if not chunks:
        return
    livres = list(dict.fromkeys(_joli_titre(c["titre"]) for c in chunks))
    st.caption("📚 Livres consultés : " + ", ".join(livres))
    for c in chunks:
        label = f"📄 {_joli_titre(c['titre'])}"
        if c.get("page"):
            label += f" — page {c['page']}"
        with st.expander(label):
            st.write(c["texte"])


def _afficher_points(msg: dict) -> None:
    """Rend un message 'points clés de chapitre' dans la conversation, avec pour
    chaque point ses passages sources dépliables (texte réellement stocké en base)."""
    st.markdown(f"### 📌 Points clés — {_joli_titre(msg['livre'])} "
                f"(pages {msg['page_debut']}–{msg['page_fin']})")
    points = msg["points"]
    if isinstance(points, str):
        st.write(points)
    else:
        for i, p in enumerate(points, start=1):
            st.markdown(f"**{i}. {p['titre']}**")
            st.write(p["explication"])
            passages = _passages_pour_pages(msg["chunks"], p["pages"])
            with st.expander(f"📄 Passages associés (pages {p['pages']})"):
                if not passages:
                    st.caption("Passage exact non retrouvé sur ces pages.")
                else:
                    for c in passages:
                        st.markdown(f"*Page {c['page']}*")
                        st.write(c["texte"])
                        st.markdown("---")
    st.caption(f"Basé sur {len(msg['chunks'])} passages du livre.")


st.set_page_config(page_title="BookMind", page_icon="📚")
st.title("📚 BookMind — Discute avec ta bibliothèque")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Barre latérale : entretien + outil chapitre ───────────────────────────────
st.sidebar.markdown(
    "**BookMind** répond en s'appuyant **uniquement** sur les livres de ta "
    "bibliothèque, avec raisonnement et sources citées — jamais de réponse inventée."
)

if st.session_state.messages and st.sidebar.button("🗑️ Nouvelle conversation"):
    st.session_state.messages = []
    st.rerun()

with st.sidebar.expander("📖 Points clés d'un chapitre"):
    st.caption("Choisis un livre et un chapitre : les points clés arrivent "
               "dans la conversation. (Plusieurs minutes : le chapitre entier est lu.)")
    try:
        livres = _livres_cache()
    except IndexAbsentError:
        livres = []
    if not livres:
        st.caption("Aucun livre paginé dans la base — ingère des PDF d'abord.")
    else:
        titres = [lv["titre"] for lv in livres]
        titre_sel = st.selectbox("Livre :", titres, format_func=_joli_titre)
        lv = next(l for l in livres if l["titre"] == titre_sel)
        chapitres = _chapitres_cache(titre_sel)

        if chapitres:
            options = [f"{c['titre']}  (p. {c['page_debut']}–{c['page_fin']})"
                       for c in chapitres]
            choix = st.selectbox("Chapitre :", options)
            chap = chapitres[options.index(choix)]
            page_debut, page_fin = chap["page_debut"], chap["page_fin"]
            libelle = chap["titre"]
        else:
            st.caption(f"Sommaire non détecté — indique une plage de pages "
                       f"({lv['page_min']}–{lv['page_max']}).")
            page_debut = st.number_input("Page de début", min_value=int(lv["page_min"]),
                                         max_value=int(lv["page_max"]), value=int(lv["page_min"]))
            page_fin = st.number_input("Page de fin", min_value=int(lv["page_min"]),
                                       max_value=int(lv["page_max"]), value=int(lv["page_max"]))
            libelle = f"pages {page_debut} à {page_fin}"

        sujet = st.text_input("Sujet à cibler (optionnel) :", key="sujet_chapitre")
        if st.button("📌 Extraire les points clés"):
            if page_fin < page_debut:
                st.error("Page de fin < page de début.")
            else:
                demande = (f"Points clés de « {libelle} » — {_joli_titre(titre_sel)}"
                           + (f" (sujet : {sujet})" if sujet.strip() else ""))
                with st.spinner("Lecture du chapitre (plusieurs minutes)..."):
                    try:
                        chunks_ch, points = _points_cles_cache(
                            titre_sel, int(page_debut), int(page_fin), sujet)
                        st.session_state.messages.append(
                            {"role": "user", "content": demande})
                        if not chunks_ch:
                            st.session_state.messages.append(
                                {"role": "assistant",
                                 "content": "Aucun texte trouvé sur cette plage de pages."})
                        else:
                            st.session_state.messages.append(
                                {"role": "assistant", "type": "points",
                                 "livre": titre_sel, "page_debut": int(page_debut),
                                 "page_fin": int(page_fin), "points": points,
                                 "chunks": chunks_ch})
                    except (OllamaError, IndexAbsentError, EmbeddingError) as e:
                        st.error(str(e))
                st.rerun()

if st.sidebar.button("🔄 Recharger l'index documentaire"):
    if ingestion_necessaire():
        with st.spinner("Nouveaux documents détectés — ingestion en cours "
                        "(conversion PDF, découpage, embeddings — peut prendre plusieurs minutes)..."):
            res = subprocess.run([sys.executable, str(RACINE / "ingest.py")],
                                 cwd=str(RACINE), capture_output=True, text=True)
        if res.returncode != 0:
            st.sidebar.error("Échec de l'ingestion :")
            st.sidebar.code((res.stderr or res.stdout)[-2000:])
        else:
            recharger_index()
            st.cache_data.clear()
            st.sidebar.success("Ingestion terminée — les nouveaux documents sont intégrés.")
    else:
        recharger_index()
        st.cache_data.clear()
        st.sidebar.success("Index rechargé — aucun nouveau document détecté dans data/raw/.")

# ── Conversation ──────────────────────────────────────────────────────────────
st.caption("Pose une question sur tes livres : réponse raisonnée, organisée et sourcée, "
           "basée uniquement sur ta bibliothèque. Compte 1 à 3 min par réponse sur ce PC "
           "(génération et raisonnement 100 % locaux).")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "points":
            _afficher_points(msg)
        else:
            st.write(msg["content"])
            if msg["role"] == "assistant":
                _afficher_sources(msg.get("sources", []))

if question := st.chat_input("Pose ta question sur tes livres..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        # Historique = tout sauf la question qu'on vient d'ajouter (les messages
        # « points clés » sont résumés par leur titre pour ne pas gonfler le prompt)
        historique = []
        for m in st.session_state.messages[:-1]:
            if m.get("type") == "points":
                historique.append(("assistant",
                                   f"[Points clés du livre {_joli_titre(m['livre'])}, "
                                   f"pages {m['page_debut']}-{m['page_fin']}]"))
            else:
                historique.append((m["role"], m["content"]))
        # Question de suivi courte (« et comment le pratiquer ? ») : seule, elle
        # ne retrouve rien — on l'enrichit de la question précédente pour la recherche.
        requete = question
        if historique and len(question.split()) < 7:
            derniere_q = next((t for r, t in reversed(historique) if r == "user"), "")
            requete = f"{derniere_q} {question}"

        chunks = []
        try:
            with st.spinner("Recherche dans toute la bibliothèque..."):
                chunks = rechercher(requete)
            with st.spinner("Réflexion et rédaction de la réponse (1 à 3 min sur ce PC)..."):
                reponse = repondre(question, chunks, historique=historique)
        except IndexAbsentError as e:
            reponse, chunks = str(e), []
        except (OllamaError, EmbeddingError) as e:
            reponse, chunks = f"⚠️ {e}", []
        st.write(reponse)
        _afficher_sources(chunks)

    st.session_state.messages.append(
        {"role": "assistant", "content": reponse, "sources": chunks})
