import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import base64


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )


data = pd.read_csv('cleaned_data.csv')
data = pd.DataFrame(data)
model = pickle.load(open('model.pkl', 'rb'))


st.title('House Price Predictor')
st.subheader('Use this tool to predict the price for a house in Bangalore!')

location = st.selectbox(
    'Select the Location:',
    np.unique(data['location'].values)
)

bhk = st.slider("Select the number of BHK", 1, 20)

bath = st.slider("Select the number of Bathrooms", 1, 20)

sqft = st.text_input('Enter the Square Feet:')

pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])


if st.button('Predict'):
    prediction = np.round((model.predict(pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk']))[0])*100000, 2)
    st.subheader("Your house will cost you around:")
    st.subheader("\u20B9" + str(prediction))



