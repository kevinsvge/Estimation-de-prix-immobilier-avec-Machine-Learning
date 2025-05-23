import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_data(path, test_size=0.2):
    # 🔹 Lecture du fichier CSV contenant les données
    df = pd.read_csv(path)
    
    # 🔹 Suppression de colonnes peu utiles ou redondantes
    df = df.drop(columns=["Id", "Utilities", "Street"])

    # 🔹 Mapping des variables qualitatives ordinales (qualité / état)
    # Remplace les valeurs textuelles par des scores numériques
    qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "Missing": 0}
    for col in ["ExterQual", "ExterCond", "KitchenQual", "HeatingQC"]:
        df[col] = df[col].fillna("Missing").map(qual_map)

    # 🔹 Encodage one-hot pour les autres variables catégorielles (avec drop_first pour éviter la multicolinéarité)
    df = pd.get_dummies(df, drop_first=True)

    # 🔹 Suppression des biens les plus chers (top 1%) pour réduire les outliers
    df = df[df["SalePrice"] < df["SalePrice"].quantile(0.99)]

    # 🔹 Séparation des variables explicatives (X) et de la variable cible (y)
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    # 🔹 Découpage du jeu de données en jeu d'entraînement et de test
    return train_test_split(X, y, test_size=test_size, random_state=42)
