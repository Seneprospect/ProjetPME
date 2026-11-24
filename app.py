import io
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from src.etl import ALIASES_DATE, ALIASES_MONTANT, trouver_colonne, executer_etl
from src.model_ml import entrainer_et_predire

st.set_page_config(
    page_title="Prévision Ventes PME — Data & ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

INPUT_FILE = "data/Ventes_PME_Senegal.xlsx"
OUTPUT_FILE = "data/Previsions_Ventes_ML.xlsx"

# --- STYLE CSS SOMBRE & BLEU PROFESSIONNEL ---
st.markdown(
    """
    <style>
        /* Arrière-plan général bleu nuit profond */
        .stApp {
            background: linear-gradient(180deg, #070E1A 0%, #0D1B2A 100%);
            color: #E2E8F0;
        }

        /* Barre latérale bleu marine */
        [data-testid="stSidebar"] {
            background-color: #0A1628;
            border-right: 1px solid rgba(56, 189, 248, 0.18);
        }

        /* Bannière d'en-tête */
        .hero-box {
            background: linear-gradient(135deg, #102542 0%, #16324F 60%, #1B4965 100%);
            border: 1px solid rgba(56, 189, 248, 0.28);
            border-radius: 14px;
            padding: 20px 26px;
            margin-bottom: 18px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.35);
        }
        .hero-title {
            font-size: 1.75rem;
            font-weight: 700;
            color: #F8FAFC;
            margin: 0 0 6px 0;
        }
        .hero-sub {
            font-size: 0.95rem;
            color: #93C5FD;
            margin: 0;
        }

        /* Cartes KPI personnalisées */
        .kpi-card {
            background: rgba(16, 37, 66, 0.78);
            border: 1px solid rgba(56, 189, 248, 0.22);
            border-left: 4px solid #38BDF8;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
        }
        .kpi-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #94A3B8;
            margin-bottom: 6px;
        }
        .kpi-value {
            font-size: 1.45rem;
            font-weight: 700;
            color: #F8FAFC;
            margin: 0;
        }
        .kpi-note {
            font-size: 0.78rem;
            color: #7DD3FC;
            margin-top: 6px;
        }

        /* Bouton principal bleu professionnel */
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #0284C7 0%, #2563EB 100%);
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 4px 14px rgba(56, 189, 248, 0.4);
            transform: translateY(-1px);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def generer_template_excel() -> bytes:
    """Génère un modèle Excel simple avec exemples pour les PME."""
    df_template = pd.DataFrame(
        {
            "Date": ["2025-01-15", "2025-01-20", "2025-02-10"],
            "Montant_Total_FCFA": [450000, 320000, 610000],
        }
    )
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_template.to_excel(writer, index=False, sheet_name="Ventes")
    return buffer.getvalue()


def format_fcfa(valeur: float) -> str:
    return f"{valeur:,.0f} FCFA".replace(",", " ")


# --- BARRE LATÉRALE : GESTION DES DONNÉES & PARAMÈTRES ---
with st.sidebar:
    st.markdown("### 🎛️ Pilotage & Données")

    st.download_button(
        label="📄 Télécharger le modèle Excel",
        data=generer_template_excel(),
        file_name="Modele_Ventes_PME.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Importer un fichier de ventes (.xlsx)",
        type=["xlsx"],
        help="Importez votre fichier brut : vous pourrez confirmer ou ajuster les colonnes ci-dessous.",
    )

    col_date_choisie = None
    col_montant_choisi = None

    if uploaded_file is not None:
        try:
            df_apercu = pd.read_excel(uploaded_file).dropna(how="all")
            colonnes_dispos = list(df_apercu.columns)

            if colonnes_dispos:
                st.markdown("**Mapping des colonnes**")

                auto_date = trouver_colonne(df_apercu, ALIASES_DATE)
                auto_montant = trouver_colonne(df_apercu, ALIASES_MONTANT)

                idx_date = colonnes_dispos.index(auto_date) if auto_date in colonnes_dispos else 0
                idx_montant = (
                    colonnes_dispos.index(auto_montant)
                    if auto_montant in colonnes_dispos
                    else (1 if len(colonnes_dispos) > 1 else 0)
                )

                col_date_choisie = st.selectbox(
                    "Colonne de date",
                    options=colonnes_dispos,
                    index=idx_date,
                )
                col_montant_choisi = st.selectbox(
                    "Colonne du montant (FCFA)",
                    options=colonnes_dispos,
                    index=idx_montant,
                )

                if col_date_choisie == col_montant_choisi:
                    st.warning("Choisissez deux colonnes distinctes.")
        except Exception as e:
            st.error(f"Lecture du fichier impossible : {e}")

    horizon_mois = st.slider(
        "Horizon de prévision (mois)",
        min_value=1,
        max_value=6,
        value=3,
    )

    lancer = st.button("🚀 Mettre à jour & Exécuter", type="primary", use_container_width=True)

# --- EN-TÊTE PRINCIPAL ---
st.markdown(
    """
    <div class="hero-box">
        <p class="hero-title">📈 Pilotage & Prévision des Ventes PME</p>
        <p class="hero-sub">Pipeline ETL Python • Régression Linéaire • Synchronisation Power BI</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- EXÉCUTION DU PIPELINE ---
if lancer:
    try:
        with st.spinner("Validation, nettoyage et entraînement du modèle..."):
            if uploaded_file is not None:
                if not col_date_choisie or not col_montant_choisi:
                    st.error("Veuillez sélectionner la colonne de date et la colonne de montant.")
                    st.stop()
                if col_date_choisie == col_montant_choisi:
                    st.error("La colonne de date et la colonne de montant doivent être différentes.")
                    st.stop()

                df_upload = pd.read_excel(uploaded_file).dropna(how="all").copy()
                df_normalise = pd.DataFrame(
                    {
                        "Date": df_upload[col_date_choisie],
                        "Montant_Total_FCFA": df_upload[col_montant_choisi],
                    }
                )
                df_normalise.to_excel(INPUT_FILE, index=False)
                st.toast("Colonnes mappées et fichier source mis à jour !", icon="📥")

            df_mensuel = executer_etl(INPUT_FILE)
            df_final = entrainer_et_predire(df_mensuel, horizon_mois=horizon_mois)
            df_final.to_excel(OUTPUT_FILE, index=False)

            st.session_state["derniere_maj"] = datetime.now().strftime("%d/%m/%Y à %H:%M:%S")
        st.success(f"Pipeline exécuté avec succès • Prévisions générées sur {horizon_mois} mois.")
    except Exception as e:
        st.error(f"Erreur lors de l'exécution du pipeline : {e}")

# --- AFFICHAGE DES RÉSULTATS ---
try:
    df_visu = pd.read_excel(OUTPUT_FILE)
    df_visu["Date"] = pd.to_datetime(df_visu["Date"])
    df_visu["Mois"] = df_visu["Date"].dt.strftime("%Y-%m")

    reel = df_visu[df_visu["Type_Donnee"] == "Réel"].sort_values("Date")
    pred = df_visu[df_visu["Type_Donnee"] == "Prédiction (ML)"].sort_values("Date")

    dernier_mois_str = reel["Mois"].iloc[-1] if not reel.empty else "N/A"
    maj_str = st.session_state.get("derniere_maj", "Fichier existant chargé")

    st.caption(f"📅 Dernier mois historique : **{dernier_mois_str}**   |   🕒 Mise à jour : **{maj_str}**")

    # Cartes KPI stylisées
    ca_total = reel["Montant_Total_FCFA"].sum() if not reel.empty else 0.0
    dernier_ca = reel["Montant_Total_FCFA"].iloc[-1] if not reel.empty else 0.0
    prev_moy = pred["Montant_Total_FCFA"].mean() if not pred.empty else 0.0

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">CA Réel Cumulé</div>
                <div class="kpi-value">{format_fcfa(ca_total)}</div>
                <div class="kpi-note">{len(reel)} mois d'historique consolidé</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left-color: #60A5FA;">
                <div class="kpi-label">Dernier Mois Réel ({dernier_mois_str})</div>
                <div class="kpi-value">{format_fcfa(dernier_ca)}</div>
                <div class="kpi-note">Clôture mensuelle vérifiée</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left-color: #F59E0B;">
                <div class="kpi-label">Prévision Moyenne ({len(pred)} mois)</div>
                <div class="kpi-value">{format_fcfa(prev_moy)}</div>
                <div class="kpi-note">Projection Régression Linéaire</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Graphique Altair assorti au thème bleu/sombre
    st.markdown("### 📊 Trajectoire des Ventes : Réel vs Prédiction (ML)")

    chart = (
        alt.Chart(df_visu.sort_values("Date"))
        .mark_line(point=alt.OverlayMarkDef(size=65, filled=True), strokeWidth=3)
        .encode(
            x=alt.X(
                "Mois:N",
                sort=None,
                title="Mois",
                axis=alt.Axis(labelColor="#CBD5E1", titleColor="#93C5FD", gridColor="rgba(148,163,184,0.12)"),
            ),
            y=alt.Y(
                "Montant_Total_FCFA:Q",
                title="Montant (FCFA)",
                axis=alt.Axis(labelColor="#CBD5E1", titleColor="#93C5FD", gridColor="rgba(148,163,184,0.12)"),
            ),
            color=alt.Color(
                "Type_Donnee:N",
                title="Série",
                scale=alt.Scale(
                    domain=["Réel", "Prédiction (ML)"],
                    range=["#38BDF8", "#FBBF24"],
                ),
                legend=alt.Legend(labelColor="#E2E8F0", titleColor="#93C5FD"),
            ),
            tooltip=[
                alt.Tooltip("Mois:N", title="Mois"),
                alt.Tooltip("Type_Donnee:N", title="Type"),
                alt.Tooltip("Montant_Total_FCFA:Q", title="Montant (FCFA)", format=",.0f"),
            ],
        )
        .properties(height=390)
        .configure_view(strokeOpacity=0, fill="rgba(16, 37, 66, 0.45)")
        .configure(background="transparent")
    )

    st.altair_chart(chart, use_container_width=True)

    with st.expander("📋 Consulter ou télécharger les données générées"):
        st.dataframe(
            df_visu[["Mois", "Type_Donnee", "Montant_Total_FCFA"]].sort_values("Mois"),
            use_container_width=True,
            hide_index=True,
        )
        with open(OUTPUT_FILE, "rb") as f:
            st.download_button(
                label="⬇️ Télécharger Previsions_Ventes_ML.xlsx",
                data=f,
                file_name="Previsions_Ventes_ML.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

except FileNotFoundError:
    st.warning("Aucun fichier de prévision trouvé. Cliquez sur **Mettre à jour & Exécuter** dans la barre latérale.")