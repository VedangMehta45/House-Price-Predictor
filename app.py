import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from huggingface_hub import hf_hub_download
import os

st.set_page_config(page_title= "House Price Predictor", layout="centered")

@st.cache_resource  # caches on server so model loads only once
def load_model_from_hf():
    # downloads to HF cache and returns local path
    model_path = hf_hub_download(
        repo_id="vedangmehta/house-price-predictor",
        filename="house_price_pipeline.joblib",
        repo_type="model"  # optional
        # token: optional, if repo is private set `token=YOUR_TOKEN` or set env var below
    )
    return joblib.load(model_path)

model = load_model_from_hf()



# Load pipeline
# model = joblib.load('house_price_pipeline_py37_new.joblib')


#Filtered data for unique columns for location dropdown: 
data = pd.read_csv("Cleaned_data_BLR_house.csv")
unique_locations = sorted(data["location"].unique())  


page_bg = """
<style>

/* Add semi-transparent overlay */
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    # background: #fdf5e6; /* white overlay with opacity */
    background-color: #fdf5e6
    z-index: -1;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


st.title("House Price Prediction App")
st.markdown("Welcome to the House Price Predictor app! Enter house details and get price prediction (in lakhs rupees).")

# Sidebar for inputs
st.sidebar.header("Input Features")

# Collect user inputs
location = st.sidebar.selectbox("Select Location", unique_locations)
total_sqft = st.sidebar.number_input("Total Sqft", min_value=400, max_value=10000, step=100)
bath = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
bhk = st.sidebar.number_input("Number of BHK", min_value=1, max_value=10, step=1)
price_per_sqft_in_thousand = total_sqft / 1000 


# Prepare DataFrame
input_data = pd.DataFrame({
    "location": [location],
    "total_sqft": [total_sqft],
    "bath": [bath],
    "BHK": [bhk],
    "price_per_sqft_in_thousand": [price_per_sqft_in_thousand]
})

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f" Estimated Price: **{round(prediction,2)} Lakhs**")


#Footer section

st.markdown("---")

st.markdown(
    """
    <div style="position: fixed; bottom: 0; left: 0; width: 100%; 
                background-color: #f9f9f9; padding: 10px; 
                text-align: center; font-size: 14px; color: #333; 
                border-top: 1px solid #ddd;">
        Made with ❤️ | 📧 Contact:
        <a href="mailto:vedangmehta07@gmail.com" 
           style="text-decoration: none; color: #1f77b4;">
           vedangmehta07@gmail.com
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
