# 🏡 House Price Predictor  

A Machine Learning web application that predicts house prices based on key features such as location, size, and amenities.  
The model is deployed with **Streamlit** for an interactive frontend and trained using multiple regression algorithms.  

## 🚀 Tech Stack
- **Python** (pandas, numpy, matplotlib, seaborn)
- **Scikit-learn** (Linear Regression, Random Forest, XGBoost)
- **Joblib** (model serialization)
- **Streamlit** (frontend UI)

## 📂 Project Structure
- `House_Price.ipynb` → Model training and evaluation notebook  
- `Cleaned_data_BLR_house.csv` → Dataset used for training  
- `app.py` → Streamlit frontend application  
- `requirements.txt` → Project dependencies  

## ⚡ Features
- Exploratory Data Analysis & preprocessing  
- Model training with multiple algorithms  
- Interactive UI for real-time price prediction  

## 🛠️ How to Run Locally
```bash
# Clone the repo
git clone https://github.com/your-username/house-price-predictor.git
cd house-price-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
