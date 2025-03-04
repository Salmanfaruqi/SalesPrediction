from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import pickle
import streamlit as st
import pandas as pd 
# import matplotlib.pyplot as plt

def load_model():
    with open("poly_regression_model.pkl", 'rb') as file:
        model =  pickle.load(file)
    return model

def preprocessing_input_data(data):
    df = pd.DataFrame([data])
    return df

def predict_data(data):
    model = load_model()
    processed_data = preprocessing_input_data(data)
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.title("Icrecream Sales Prediction")
    st.write("Enter the Temperature for Sales Prediction")

    
    temp = st.number_input("Temperature in Celcius", min_value=-20, max_value=55, value=0)

    
    if st.button("Predict your Sales"):
       user_data = {
           "Temperature (°C)": temp
       }
       prediction = predict_data(user_data)
       st.success(f"Your Prediction Sales result is {prediction}")

if __name__ == "__main__":
    main()








