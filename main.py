from datetime import datetime
from src.etl import executer_etl
from src.model_ml import entrainer_et_predire


def run_pipeline():
    debut = datetime.now()
    print(f"[{debut.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 Démarrage du pipeline Data & ML PME...")

    input_file = "data/Ventes_PME_Senegal.xlsx"
    output_file = "data/Previsions_Ventes_ML.xlsx"

    df_mensuel = executer_etl(input_file)
    df_final = entrainer_et_predire(df_mensuel, horizon_mois=3)

    print("⏳ [3/3] Export des données vers Excel pour Power BI...")
    df_final.to_excel(output_file, index=False)

    duree = (datetime.now() - debut).total_seconds()
    print(f"✅ Pipeline terminé avec succès en {duree:.2f}s -> {output_file}")


if __name__ == "__main__":
    run_pipeline()