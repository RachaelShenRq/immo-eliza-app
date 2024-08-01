import streamlit as st
import pandas as pd
import pickle

# Load the preprocessor and model
preprocessor = pickle.load(open("data/preprocessor.pkl", "rb"))
model = pickle.load(open("data/random_forest.pkl", "rb"))

# Function to get user input
def user_input_features():
    property_types = {
        'House': 1,
        'Apartment': 2
    }

    st.markdown("<h1 style='text-align: center; color: black;'>House Price Prediction in Belgium</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: grey;'>Please enter the details below</h3>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        TypeOfProperty = st.selectbox('Type of Property', list(property_types.keys()))
        TypeOfSale = st.selectbox('Type of Sale', ['Buy', 'Rent'])
        postal_code = st.text_input('Postal Code', '0000')
        rooms = st.number_input('Number of Rooms', min_value=1, max_value=20, value=1)
        facade = st.number_input('Number of Facades', min_value=1, max_value=4, value=2)
        bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=5, value=2)
        property_condition = st.selectbox('Property Condition', ['Poor', 'Average', 'Good'])
        kitchen_equipped = st.selectbox('Kitchen Equipped', ['No', 'Yes'])
        garden = st.selectbox('Garden', ['No', 'Yes'])

    with col2:
        year_built = st.text_input('Year Built', '1980')
        stories = st.number_input('Number of Stories', min_value=1, max_value=4, value=2)
        lot_area = st.text_input('Lot Area (m²)', '50')
        LivingArea = st.text_input('Living Area (m²)', '70')
        basement = st.selectbox('Basement', ['No', 'Yes'])
        garage = st.selectbox('Garage', ['No', 'Yes'])
        garage_cars = 0
        if garage == 'Yes':
            garage_cars = st.number_input('Number of Cars Garage Can Accommodate', min_value=0, max_value=4, value=2, step=1)
        heating_cooling = st.selectbox('Heating/Cooling System', ['No', 'Yes'])

    TypeOfSale = 1 if TypeOfSale == 'Buy' else 2
    TypeOfProperty = property_types[TypeOfProperty]

    data = {
        'TypeOfProperty': TypeOfProperty,
        'TypeOfSale': TypeOfSale,
        'BedroomCount': rooms,
        'LivingArea': float(LivingArea),
        'PostalCode': int(postal_code),
        'NumberOfFacades': facade,
        'BathroomCount': bathrooms,
        'SurfaceOfPlot': float(lot_area),
        'YearBuilt': int(year_built),
        'NumberOfStories': stories,
        'BasementArea': 0 if basement == 'No' else 1,  # Assuming BasementArea to be 1 if basement is present
        'GarageSize': 0 if garage == 'No' else garage_cars,
        'HeatingCoolingSystem': 1 if heating_cooling == 'Yes' else 0,
        'PropertyCondition': 1 if property_condition == 'Average' else (2 if property_condition == 'Good' else 0),
        'KitchenEquipped': 1 if kitchen_equipped == 'Yes' else 0,
        'Garden': 1 if garden == 'Yes' else 0,
        'GardenArea': 0 if garden == 'No' else float(lot_area) - float(LivingArea)  # Assuming GardenArea to be lot_area - LivingArea if garden is present
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Apply the preprocessor and model to the user input
input_preprocessed = preprocessor.transform(input_df)
prediction = model.predict(input_preprocessed)

# Display the prediction with styling
if st.button('Predict Price'):
    st.markdown("<h2 style='text-align: center; color: red;'>Prediction</h2>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='color: red; text-align: center; font-size: 3rem;'>The estimated price is: €{prediction[0]:,.2f}</h1>", unsafe_allow_html=True)
