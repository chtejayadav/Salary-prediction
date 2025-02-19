import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pickle import load

# Load dataset
data = pd.read_csv("sd.csv")
X = data[['YearsExperience']]
Y = data['Salary']

# Split data for modeling
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Load trained Linear Regression model
loaded_model = load(open('sd', 'rb'))

def salary_prediction(input_data):
    """Predicts salary based on years of experience."""
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_array)
    return round(prediction[0], 2)

def main():
    st.title('Salary Prediction based on Years of Experience')
    
    # Collect user input
    years_experience = st.number_input('Years of Experience', min_value=0.0, format="%.1f")
    
    prediction = ''
    if st.button('Predict Salary'):
        prediction = salary_prediction([years_experience])
        st.success(f'Predicted Salary: ${prediction}')
    
if __name__ == '__main__':
    main()
