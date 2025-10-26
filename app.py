
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load the saved model and scaler
model = tf.keras.models.load_model('iris_model.h5')
scaler = joblib.load('scaler.joblib')

# Set page title
st.title('Iris Flower Classification')

# Create input fields for features
st.header('Enter Flower Measurements')
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=0.2)

# Create a button for prediction
if st.button('Predict Iris Type'):
    # Prepare input data
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    # Scale features
    features_scaled = scaler.transform(features)
    # Make prediction
    prediction = model.predict(features_scaled)
    predicted_class = np.argmax(prediction)
    
    # Map class index to iris type
    iris_types = ['setosa', 'versicolor', 'virginica']
    predicted_type = iris_types[predicted_class]
    
    # Display prediction with probability
    st.success(f'Predicted Iris Type: {predicted_type.capitalize()}')
    st.write('Prediction Probabilities:')
    for iris_type, prob in zip(iris_types, prediction[0]):
        st.write(f'{iris_type.capitalize()}: {prob:.4f}')
