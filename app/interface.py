import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Estimation immobilière", layout="wide")
st.title("🏡 Estimation du prix immobilier")

# Choix du modèle
model_choice = st.radio("Choisissez le modèle :", ["XGBoost", "Random Forest"])
model_path = "models/best_xgboost_model.joblib" if model_choice == "XGBoost" else "models/best_random_forest_model.joblib"

# Chargement du modèle
if not os.path.exists(model_path):
    st.error(f"❌ Modèle {model_choice} introuvable. Veuillez d'abord entraîner et sauvegarder le modèle.")
    st.stop()

model = joblib.load(model_path)

st.subheader("📝 Remplissez les informations du bien")

col1, col2, col3 = st.columns(3)

with col1:
    gr_liv_area = st.slider("Surface habitable (en pieds²)", 500, 5000, 1500, step=50)
    year_built = st.slider("Année de construction", 1870, 2024, 1990)
    first_flr_sf = st.slider("Surface 1er étage (pieds²)", 0, 3000, 1000)
    second_flr_sf = st.slider("Surface 2e étage (pieds²)", 0, 2000, 500)
    total_bsmt_sf = st.slider("Surface totale du sous-sol (pieds²)", 0, 3000, 800)
    garage_area = st.slider("Surface garage (pieds²)", 0, 1500, 400)

with col2:
    overall_qual = st.selectbox("Qualité générale", list(range(1, 11)), index=5)
    overall_cond = st.selectbox("État général", list(range(1, 11)), index=5)
    full_bath = st.slider("Salles de bain complètes", 0, 4, 2)
    half_bath = st.slider("Salles de bain partielles", 0, 2, 1)
    bedroom_abv_gr = st.slider("Chambres au-dessus du sol", 0, 10, 3)
    tot_rms_abv_grd = st.slider("Nombre total de pièces", 2, 15, 6)

with col3:
    kitchen_qual = st.selectbox("Qualité cuisine", ["Ex", "Gd", "TA", "Fa", "Po"])
    exter_qual = st.selectbox("Qualité extérieure", ["Ex", "Gd", "TA", "Fa", "Po"])
    exter_cond = st.selectbox("État extérieur", ["Ex", "Gd", "TA", "Fa", "Po"])
    heating_qc = st.selectbox("Qualité chauffage", ["Ex", "Gd", "TA", "Fa", "Po"])
    fireplaces = st.slider("Nombre de cheminées", 0, 3, 1)
    garage_cars = st.slider("Nombre de voitures dans garage", 0, 4, 2)

input_data = pd.DataFrame([{
    "GrLivArea": gr_liv_area,
    "YearBuilt": year_built,
    "1stFlrSF": first_flr_sf,
    "2ndFlrSF": second_flr_sf,
    "TotalBsmtSF": total_bsmt_sf,
    "GarageArea": garage_area,
    "OverallQual": overall_qual,
    "OverallCond": overall_cond,
    "FullBath": full_bath,
    "HalfBath": half_bath,
    "BedroomAbvGr": bedroom_abv_gr,
    "TotRmsAbvGrd": tot_rms_abv_grd,
    "KitchenQual": kitchen_qual,
    "ExterQual": exter_qual,
    "ExterCond": exter_cond,
    "HeatingQC": heating_qc,
    "Fireplaces": fireplaces,
    "GarageCars": garage_cars
}])

# Conversion des colonnes qualitatives
qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
for col in ["KitchenQual", "ExterQual", "ExterCond", "HeatingQC"]:
    if col in input_data:
        input_data[col] = input_data[col].map(qual_map)

if st.button("🔍 Estimer le prix"):
    try:
        template = pd.read_csv("models/template_columns.csv")
        input_data_full = pd.DataFrame([0]*len(template.columns), index=template.columns).T
        input_data_full.columns = template.columns
        for col in input_data.columns:
            input_data_full[col] = input_data[col].values[0]
        prediction = model.predict(input_data_full)[0]
        st.success(f"💰 Prix estimé avec {model_choice} : {round(prediction, 2)} €")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")