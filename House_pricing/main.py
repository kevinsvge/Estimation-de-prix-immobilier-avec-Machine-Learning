from src.data_preparation import load_and_prepare_data 
from src.model_training import test_multiple_models
from src.model_optimization import optimize_random_forest
from src.model_xgboost import optimize_model_comparison

def main():
    X_train, X_test, y_train, y_test = load_and_prepare_data("data/train.csv")

    # Évaluer plusieurs modèles de base
    test_multiple_models(X_train, X_test, y_train, y_test)

    # Optimiser Random Forest
    best_model = optimize_random_forest(X_train, y_train, X_test, y_test)

    # Prédiction test finale CPU
    y_pred = best_model.predict(X_test)
    print("\n✅ Prédictions faites avec le meilleur modèle optimisé.")

    # Optimisation & entraînement Random Forest + XGBoost GPU
    best_rf, best_xgb = optimize_model_comparison(X_train, y_train, X_test, y_test)

    print("\n✅ Modèles optimisés et enregistrés avec succès :")
    print("📁 Random Forest : best_random_forest_model.joblib")
    print("📁 XGBoost GPU   : best_xgboost_model.joblib")
    

if __name__ == "__main__":
    main()
