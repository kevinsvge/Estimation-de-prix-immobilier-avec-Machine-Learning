# 📦 Imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from math import sqrt


def build_preprocessing_pipeline(X):
    # Séparation des colonnes numériques et catégorielles
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    # Pipeline pour les colonnes numériques : imputation + standardisation
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Pipeline pour les colonnes catégorielles : imputation + encodage
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combinaison des deux pipelines dans un ColumnTransformer
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    return preprocessor



def optimize_random_forest(X_train, y_train, X_test, y_test):
    # Construction du pipeline de prétraitement
    preprocessor = build_preprocessing_pipeline(X_train)

    # Pipeline complet : prétraitement + modèle Random Forest
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ])

    # Grille de recherche des hyperparamètres
    param_grid = {
        "model__n_estimators": [100, 200],           # nombre d'arbres
        "model__max_depth": [10, 20, None],          # profondeur maximale
        "model__min_samples_split": [2, 5, 10]       # taille min. pour un split
    }

    # Recherche par validation croisée
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,                              # 3-fold cross-validation
        scoring="neg_mean_absolute_error", # on cherche à minimiser MAE
        n_jobs=-1,                         # utilise tous les cœurs CPU
        verbose=2
    )
    grid_search.fit(X_train, y_train)

    print("\n✅ Meilleurs hyperparamètres :", grid_search.best_params_)
    print("📉 Meilleur score MAE (négatif car inverse) :", grid_search.best_score_)

    # Évaluation finale sur le jeu test
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\n📊 Évaluation sur le jeu de test :")
    print("MAE  :", mean_absolute_error(y_test, y_pred))
    print("RMSE :", sqrt(mean_squared_error(y_test, y_pred)))
    print("R²   :", r2_score(y_test, y_pred))

    # Graphique y_test vs y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.xlabel("Prix réel")
    plt.ylabel("Prix prédit")
    plt.title("📈 Prédiction vs Réalité")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Sauvegarde du modèle optimisé
    joblib.dump(best_model, "models/best_random_forest_model.joblib")
    print("\n📁 Modèle sauvegardé sous best_random_forest_model.joblib")

    return best_model
