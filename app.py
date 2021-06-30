import pandas as pd
import numpy as np
import streamlit as st
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def main():

    df = load_data()
    df = preprocess_data(df)

    st.title('House Price Prediction for Silicon Valley of India - Bangalore')
    st.markdown('Just Enter the following details and we will predict the price of your **Dream House**')
    st.sidebar.title('Developer\'s Contact')
    st.sidebar.markdown('[![Harsh-Dhamecha]'
                        '(https://img.shields.io/badge/Author-Vaibhav%20Sharma-brightgreen)]'
                        '(https://www.linkedin.com/in/vaibhavsharma16/)')

    st.warning('Only Enter Numeric Values in the Following Fields')
    bhk = st.text_input("Total BHK")
    area = st.text_input("Area in Square Feet")
    baths = st.text_input("Total Bathrooms")
    balcony = st.selectbox("Total Balcony", ['0', '1', '2', '3'])
    submit = st.button('Predict Price')

    if submit: 
        if bhk and area and baths and balcony:
            with st.spinner('Predicting...'):
                time.sleep(2)
                bhk, area, baths, balcony = int(bhk), int(area), int(baths), int(balcony)
                x_test = np.array([[bhk, area, baths, balcony]])
                prediction = predict(df, x_test)
                st.info(f"Your **Dream House** Price is {prediction} lacs")
        else:
            st.error('Please Enter All the Details')


@st.cache
def train_model(df):
    global scaler
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
    model = LinearRegression().fit(X_train, y_train)
    return model, scaler


def predict(df, x_test):
    model, scaler = train_model(df)
    X_test = scaler.transform(x_test)
    y_pred = model.predict(X_test)
    return round(y_pred[0], 2)


def load_data():
    return pd.read_csv('Bengaluru_House_Data.csv')


def preprocess_data(df):
    df = df.loc[:, ['size', 'total_sqft', 'bath', 'balcony', 'price']]
    df.dropna(inplace=True)
    df = df[df['size'].str.contains('BHK', na=False)]
    df['size'] = df['size'].str.replace(r'\D', '').astype(int)
    df['total_sqft'] = df['total_sqft'].str.extract(r'(\d+)', expand=False)
    df['bath'] = df['bath'].astype(int)
    df['balcony'] = df['balcony'].astype(int)
    df['total_sqft'] = df['total_sqft'].astype(int)
    return df


if __name__ == '__main__':
    main()
