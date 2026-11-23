import re
import pandas as pd

ALIASES_DATE = ["date", "date_vente", "date vente", "date_facture", "date facture", "jour", "mois"]
ALIASES_MONTANT = [
    "montant_total_fcfa", "montant total fcfa", "montant_total", "montant total",
    "montant", "total", "ca", "chiffre_affaires", "chiffre affaires", "vente", "ventes", "prix"
]

def trouver_colonne(df: pd.DataFrame, aliases: list[str]) -> str | None:
    cols_norm = {col.strip().lower(): col for col in df.columns}
    for alias in aliases:
        if alias in cols_norm:
            return cols_norm[alias]
    # Recherche partielle si un mot-clé est présent (ex: "Montant (FCFA)")
    for col_low, col_orig in cols_norm.items():
        if any(alias in col_low for alias in aliases):
            return col_orig
    return None

def _nettoyer_montant(serie: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce")
    cleaned = (
        serie.astype(str)
        .str.replace(r"[^\d,.-]", "", regex=True)  # enlève 'FCFA', espaces, symboles
        .str.replace(" ", "", regex=False)
    )
    # Si présence de virgule sans point, traiter la virgule comme décimale
    cleaned = cleaned.apply(lambda v: v.replace(",", ".") if ("," in v and "." not in v) else v)
    return pd.to_numeric(cleaned, errors="coerce")

def executer_etl(input_path: str) -> pd.DataFrame:
    print("⏳ [1/3] Chargement et normalisation automatique des données brutes...")
    df = pd.read_excel(input_path)
    df = df.dropna(how="all").copy()

    col_date = trouver_colonne(df, ALIASES_DATE)
    col_montant = trouver_colonne(df, ALIASES_MONTANT)

    if not col_date or not col_montant:
        raise ValueError(
            f"Colonnes introuvables. Colonnes détectées : {list(df.columns)}. "
            "Assurez-vous d'avoir une colonne de date et une colonne de montant."
        )

    df["Date"] = pd.to_datetime(df[col_date], errors="coerce", dayfirst=True)
    df["Montant_Total_FCFA"] = _nettoyer_montant(df[col_montant])

    df = df.dropna(subset=["Date", "Montant_Total_FCFA"])

    df["Mois_Annee"] = df["Date"].dt.to_period("M")
    df_mensuel = df.groupby("Mois_Annee")["Montant_Total_FCFA"].sum().reset_index()
    df_mensuel["Date"] = df_mensuel["Mois_Annee"].dt.to_timestamp()
    df_mensuel = df_mensuel.sort_values("Date").reset_index(drop=True)

    return df_mensuel[["Date", "Montant_Total_FCFA"]]