import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_data(path, test_size=0.2):
    df = pd.read_csv(path)
    
    # Nettoyage de base
    df = df.drop(columns=["Id","Utilities", "Street"])

    qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "Missing": 0}
    for col in ["ExterQual", "ExterCond", "KitchenQual", "HeatingQC"]:
        df[col] = df[col].fillna("Missing").map(qual_map)


    df = pd.get_dummies(df, drop_first=True)
    df = df[df["SalePrice"] < df["SalePrice"].quantile(0.99)]  # Optionnel : supprime les 1 % plus chers

    # Séparation X et y
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    # Train/test split
    return train_test_split(X, y, test_size=test_size, random_state=42)