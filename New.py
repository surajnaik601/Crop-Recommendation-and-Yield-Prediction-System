import pandas as pd
import streamlit as st
import pickle
import numpy as np
import random

# Load the pre-trained model for crop recommendation
with open('model.pkl', 'rb') as model_file:
    crop_recommendation_model = pickle.load(model_file)

with open('fertilizerRecommendation.pkl', 'rb') as model_file:
    fertilizer = pickle.load(model_file)
# Load the trained model for crop production prediction
with open('random_forest_model.pkl', 'rb') as model_file:
    crop_production_model = pickle.load(model_file)


#with open('soilLabel.pkl', 'rb') as soilLabel:
#    sencoder = pickle.load(soilLabel)
#with open('cropLabel.pkl', 'rb') as cropLabel:
#    cencoder = pickle.load(cropLabel)
# Define the lists of districts, seasons, and crops
districts = ['AHMEDNAGAR', 'AKOLA', 'AMRAVATI', 'AURANGABAD', 'BEED', 'BHANDARA', 'BULDHANA', 'CHANDRAPUR', 'DHULE', 'GADCHIROLI', 'GONDIA', 'HINGOLI', 'JALGAON', 'JALNA', 'KOLHAPUR', 'LATUR', 'MUMBAI', 'NAGPUR', 'NANDED', 'NANDURBAR', 'NASHIK', 'OSMANABAD', 'PALGHAR', 'PARBHANI', 'PUNE', 'RAIGAD', 'RATNAGIRI', 'SANGLI', 'SATARA', 'SINDHUDURG', 'SOLAPUR', 'THANE', 'WARDHA', 'WASHIM', 'YAVATMAL']
seasons = ['Autumn     ',
       'Kharif     ', 'Rabi       ', 'Summer     ',
       'Whole Year ']
crops = ['Arhar/Tur', 'Bajra', 'Banana', 'Castor seed', 'Cotton(lint)', 'Gram', 'Grapes', 'Groundnut', 'Jowar', 'Linseed', 'Maize', 'Mango', 'Moong(Green Gram)', 'Niger seed', 'Onion', 'Other  Rabi pulses', 'Other Cereals & Millets', 'Other Kharif pulses', 'Pulses total', 'Ragi', 'Rapeseed &Mustard', 'Rice', 'Safflower', 'Sesamum', 'Small millets', 'Soyabean', 'Sugarcane', 'Sunflower', 'Tobacco', 'Tomato', 'Total foodgrain', 'Urad', 'Wheat', 'other oilseeds']


# Function to get user input and make crop recommendations
def recommend_crop(N, P, k, temperature, humidity, ph, rainfall):
    input_data = [N, P, k, temperature, humidity, ph, rainfall]
    features = np.array([input_data])
    prediction = crop_recommendation_model.predict(features)
    return prediction[0]

# Function to get user input and make crop production predictions
def predict_production(area, district, season, crop):
    # Create a DataFrame with all columns and set values to 0
    columns = ['Area'] + [f'District_Name_{d}' for d in districts] + \
              [f'Season_{s}' for s in seasons] + [f'Crop_{c}' for c in crops]

    user_input = pd.DataFrame(0, columns=columns, index=[0])

    # Set the user-selected values to 1
    user_input['Area'] = area
    user_input[f'District_Name_{district}'] = 1
    user_input[f'Season_{season}'] = 1
    user_input[f'Crop_{crop}'] = 1

    # Ensure the columns match the ones used during training

    # Make predictions using the loaded model
    prediction = crop_production_model.predict(user_input)

    return prediction[0]

# Streamlit app layout
def main():
    st.title('Dashboard')
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", ["Crop Recommendation", "Crop Production Prediction"])

    if selection == "Crop Recommendation":
        crop_recommendation_page()
    elif selection == "Crop Production Prediction":
        crop_production_prediction_page()

def crop_recommendation_page():
    st.header('Crop Recommendation')
    N = st.slider('Nitrogen (N) content in soil', min_value=0, max_value=100, value=40)
    P = st.slider('Phosphorus (P) content in soil', min_value=0, max_value=100, value=50)
    k = st.slider('Potassium (K) content in soil', min_value=0, max_value=100, value=50)
    temperature = st.slider('Temperature (°C)', min_value=0, max_value=100, value=40)
    humidity = st.slider('Humidity (%)', min_value=0, max_value=100, value=20)
    ph = st.slider('pH of soil', min_value=0, max_value=14, value=7)
    rainfall = st.slider('Rainfall (mm)', min_value=0, max_value=1000, value=100)

    if st.button('Recommend Crop'):
        recommended_crop_code = recommend_crop(N, P, k, temperature, humidity, ph, rainfall)
        recommended_crop = crops[recommended_crop_code]
        st.subheader('Crop Recommendation:')
        st.write(f'The recommended crop is: {recommended_crop}')

def crop_production_prediction_page():
    st.header('Crop Production Prediction')
    area = st.slider('Area (in hectares)', min_value=0.0, max_value=10000.0, value=5000.0)
    district = st.selectbox('Select District', districts, index=0)
    season = st.selectbox('Select Season', seasons, index=0)
    crop = st.selectbox('Select Crop', crops, index=0)

    if st.button('Predict Production'):
        production_prediction = predict_production(area, district, season, crop)
        st.subheader('Production Prediction:')
        st.write(f'The predicted production is: {production_prediction:.2f} tons')

def fertilizer_recommendation_page():
    st.header('Fertilizer Recommendation')
    temperature = st.slider('Temperature (°C)', min_value=0, max_value=100, value=40)
    humidity = st.slider('Humidity (%)', min_value=0, max_value=100, value=20)
    soil_moisture = st.slider('Soil Moisture', min_value=0, max_value=100, value=50)
    soil_type = st.selectbox('Soil Type', ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey'])
    crop_type = st.selectbox('Crop Type', ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley',
       'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts'])
    nitrogen = st.slider('Nitrogen (N) content in soil', min_value=0, max_value=100, value=40)
    potassium = st.slider('Potassium (K) content in soil', min_value=0, max_value=100, value=50)
    phosphorus = st.slider('Phosphorus (P) content in soil', min_value=0, max_value=100, value=50)
    crop_values = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']
    numeric_values = [3, 8, 1, 9, 6, 0, 10, 4, 5, 7, 2]

    crop_dict = dict(zip(crop_values, numeric_values))

    soil_values = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
    numeric_values= [4, 2, 0, 3, 1]

    soil_dict = dict(zip(soil_values, numeric_values))

    if st.button('Recommend Fertilizer'):

        
        # Create a DataFrame from user input
        user_input = pd.DataFrame({
            'Temparature': [temperature],
            'Humidity': [humidity],
            'Soil Moisture': [soil_moisture],
            'Soil Type': [soil_dict[soil_type]],
            'Crop Type': [crop_dict[crop_type]],
            'Nitrogen': [nitrogen],
            'Potassium': [potassium],
            'Phosphorous': [phosphorus]
        })

        # Input the DataFrame into the fertilizer.predict function
        recommended_fertilizer = fertilizer.predict(user_input)
        st.subheader('Fertilizer Recommendation:')
        st.write(f'The recommended fertilizer is: {recommended_fertilizer}')

# ... (your existing code)

def main():
    st.title('Dashboard')
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", ["Crop Recommendation", "Crop Production Prediction", "Fertilizer Recommendation"])

    if selection == "Crop Recommendation":
        crop_recommendation_page()
    elif selection == "Crop Production Prediction":
        crop_production_prediction_page()
    elif selection == "Fertilizer Recommendation":
        fertilizer_recommendation_page()

# ... (your existing code)

# Run the Streamlit app
if __name__ == '__main__':
    main()
