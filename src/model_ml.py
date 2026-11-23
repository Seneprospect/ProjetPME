import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def entrainer_et_predire(df_mensuel: pd.DataFrame, horizon_mois: int = 3) -> pd.DataFrame:
    print("⏳ [2/3] Entraînement du modèle prédictif (Régression Linéaire)...")
    df_work = df_mensuel.copy()
    df_work['Mois_Index'] = np.arange(len(df_work))

    X = df_work[['Mois_Index']]
    y = df_work['Montant_Total_FCFA']

    model = LinearRegression()
    model.fit(X, y)

    derniere_date = df_work['Date'].max()
    futures_dates = pd.date_range(start=derniere_date + pd.DateOffset(months=1), periods=horizon_mois, freq='MS')
    futur_X = pd.DataFrame({'Mois_Index': np.arange(len(df_work), len(df_work) + horizon_mois)})

    predictions = model.predict(futur_X)

    df_predict = pd.DataFrame({
        'Date': futures_dates,
        'Montant_Total_FCFA': predictions.round(0),
        'Type_Donnee': 'Prédiction (ML)'
    })

    df_historique = df_work[['Date', 'Montant_Total_FCFA']].copy()
    df_historique['Type_Donnee'] = 'Réel'

    return pd.concat([df_historique, df_predict], ignore_index=True)