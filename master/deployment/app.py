import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="nilku/tourism_pred_model", filename="best_tourism_pred_model_v1.joblib")
#model_path = hf_hub_download(repo_id="nilku/TourismPackagePrediction", filename="best_tourism_pred_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# List of numerical features (must match prep.py)
numeric_features = [
    'Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
    'NumberOfTrips', 'MonthlyIncome', 'NumberOfChildrenVisiting'
]

# List of categorical features (must match prep.py)
categorical_features = [
    'Gender', 'MaritalStatus', 'TypeofContact', 'Occupation', 'ProductPitched',
    'Designation', 'CityTier', 'PreferredPropertyStar', 'Passport',
    'PitchSatisfactionScore', 'OwnCar'
]


# Streamlit UI for Customer Churn Prediction
st.title("Tourism Package Prediction App")
st.write("Tourism Package Prediction App is a internal tool  that predicts whether customers will taken the tourism package or not")
st.write("Kindly enter the customer details to check whether they are likely to take tourism package or not.")

# Numerical Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)
num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=6, value=2)
num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
num_trips = st.number_input("Number of Trips", min_value=0, max_value=50, value=5)
monthly_income = st.number_input("Monthly Income",min_value=0, value=50000)
num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
# Convert categorical inputs to match model training
gender = st.selectbox("Gender", ['Male', 'Female', 'Fe Male'])
marital_status = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced'])
type_of_contact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])  
product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP', 'Director'])
city_tier = st.selectbox("City Tier", [1, 2, 3])
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")



input_data = pd.DataFrame([{
        'Age': age,
        'DurationOfPitch': duration_of_pitch,
        'NumberOfPersonVisiting': num_persons,
        'NumberOfFollowups': num_followups,
        'NumberOfTrips': num_trips,
        'MonthlyIncome': monthly_income,
        'NumberOfChildrenVisiting': num_children,
        'Gender': gender,
        'MaritalStatus': marital_status,
        'TypeofContact': type_of_contact,
        'Occupation': occupation,
        'ProductPitched': product_pitched,
        'Designation': designation,
        'CityTier': city_tier,
        'PreferredPropertyStar': preferred_property_star,
        'Passport': passport,
        'PitchSatisfactionScore': pitch_satisfaction_score,
        'OwnCar': own_car
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Purchase Package" if prediction == 1 else "Decline Package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
