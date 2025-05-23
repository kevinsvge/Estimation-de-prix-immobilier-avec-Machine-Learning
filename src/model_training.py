# 📦 Imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from math import sqrt

import pandas as pd

def build_preprocessing_pipeline(X):
    # 🔹 Séparation des colonnes numériques et catégorielles
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    # 🔹 Pipeline numérique : imputation des valeurs manquantes + standardisation
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 🔹 Pipeline catégoriel : imputation + encodage one-hot
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 🔹 Fusion des deux pipelines dans un ColumnTransformer
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    return preprocessor



def evaluate_model(model_pipeline, X_train, X_test, y_train, y_test):
    # 🔹 Entraînement du pipeline complet
    model_pipeline.fit(X_train, y_train)
    
    # 🔹 Prédiction sur le jeu de test
    y_pred = model_pipeline.predict(X_test)

    # 🔹 Affichage des métriques d’évaluation
    print(f"📊 Évaluation du modèle : {model_pipeline.named_steps['model'].__class__.__name__}")
    print("MAE  :", mean_absolute_error(y_test, y_pred))
    print("RMSE :", sqrt(mean_squared_error(y_test, y_pred)))
    print("R²   :", r2_score(y_test, y_pred))
    print("------------")


def test_multiple_models(X_train, X_test, y_train, y_test):
    # 🔹 Création du pipeline de prétraitement
    preprocessor = build_preprocessing_pipeline(X_train)

    # 🔹 Dictionnaire des modèles à tester
    models = {
        "Régression Linéaire": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Arbre de Décision": DecisionTreeRegressor(random_state=42),
        "Forêt Aléatoire": RandomForestRegressor(random_state=42),
    }

    # 🔹 Pour chaque modèle : entraînement + évaluation
    for name, model in models.items():
        model_pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])
        print(f"\n🔍 Test du modèle : {name}")
        evaluate_model(model_pipeline, X_train, X_test, y_train, y_test)
