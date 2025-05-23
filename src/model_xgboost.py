# 📦 Imports nécessaires pour pipeline, optimisation et visualisation
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
import xgboost as xgb
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from math import sqrt


def build_preprocessing_pipeline(X):
    # 🔹 Sépare les colonnes numériques et catégorielles
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    # 🔹 Pipeline pour les colonnes numériques : imputation + standardisation
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 🔹 Pipeline pour les colonnes catégorielles : imputation + encodage
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 🔹 Combine les deux pipelines avec ColumnTransformer
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    return preprocessor


#🚀 Fonction d’optimisation et comparaison Random Forest / XGBoost
def optimize_model_comparison(X_train, y_train, X_test, y_test):
    preprocessor = build_preprocessing_pipeline(X_train)

    # Random Forest Pipeline
    # 🔹 Pipeline Random Forest avec prétraitement
    rf_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ])

    # 🔹 Grille d’hyperparamètres pour Random Forest
    rf_param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5, 10]
    }

    # 🔹 GridSearchCV pour optimisation
    rf_search = GridSearchCV(rf_pipeline, rf_param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=2)
    rf_search.fit(X_train, y_train)

    print("\n✅ Random Forest meilleurs hyperparamètres :", rf_search.best_params_)
    print("📉 Random Forest MAE (valeurs inversées) :", rf_search.best_score_)

    best_rf = rf_search.best_estimator_
    rf_pred = best_rf.predict(X_test)

    print("\n📊 Évaluation Random Forest :")
    print("MAE  :", mean_absolute_error(y_test, rf_pred))
    print("RMSE :", sqrt(mean_squared_error(y_test, rf_pred)))
    print("R²   :", r2_score(y_test, rf_pred))

    joblib.dump(best_rf, "models/best_random_forest_model.joblib")
    print("\n📁 Modèle Random Forest sauvegardé sous best_random_forest_model.joblib")


    # 🔹 Pipeline XGBoost avec GPU
    xgb_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", xgb.XGBRegressor(tree_method="gpu_hist", gpu_id=0, random_state=42, verbosity=0))
    ])

    # 🔹 Grille d’hyperparamètres pour XGBoost
    xgb_param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 6, 10],
        "model__learning_rate": [0.01, 0.1, 0.3]
    }

    # 🔹 GridSearchCV pour XGBoost
    xgb_search = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=2)
    xgb_search.fit(X_train, y_train)

    print("\n✅ XGBoost meilleurs hyperparamètres :", xgb_search.best_params_)
    print("📉 XGBoost MAE (valeurs inversées) :", xgb_search.best_score_)

    best_xgb = xgb_search.best_estimator_
    xgb_pred = best_xgb.predict(X_test)

    print("\n📊 Évaluation XGBoost :")
    print("MAE  :", mean_absolute_error(y_test, xgb_pred))
    print("RMSE :", sqrt(mean_squared_error(y_test, xgb_pred)))
    print("R²   :", r2_score(y_test, xgb_pred))

    joblib.dump(best_xgb, "models/best_xgboost_model.joblib")
    print("\n📁 Modèle XGBoost sauvegardé sous best_xgboost_model.joblib")


    # Comparaison graphique
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=rf_pred, label="Random Forest", alpha=0.6)
    sns.scatterplot(x=y_test, y=xgb_pred, label="XGBoost", alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label="Parfait")
    plt.xlabel("Prix réel")
    plt.ylabel("Prix prédit")
    plt.title("📊 Comparaison : RF vs XGB")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 🔄 Sauvegarde des colonnes attendues par le modèle XGBoost
    template = pd.DataFrame(columns=best_xgb.feature_names_in_)
    template.to_csv("models/template_columns.csv", index=False)
    print("\n📄 Colonnes du modèle sauvegardées dans models/template_columns.csv")


    return best_rf, best_xgb
