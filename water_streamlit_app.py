import streamlit as st
import pandas as pd
import joblib
import pickle

st.markdown(
    """
    <style>
    html, body, [data-testid="stApp"] {
        height: 100%;
        background-image: url("https://www.earth.com/assets/_next/image/?url=https%3A%2F%2Fcff2.earth.com%2Fuploads%2F2023%2F09%2F12105550%2FRiver-water-quality-1400x850.jpg&w=1200&q=75");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] > div:first-child {
        background-image: url("https://www.earth.com/assets/_next/image/?url=https%3A%2F%2Fcff2.earth.com%2Fuploads%2F2023%2F09%2F12105550%2FRiver-water-quality-1400x850.jpg&w=1200&q=75");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load trained model
st.title("Water Quality Predictor")
section = st.sidebar.selectbox("Choose Section", ["Prediction", "About"])

if section == "Prediction":
    st.info("Prediction with Random Forest Model")

    # Cache model loading
    @st.cache_resource
    def load_model():
        with open("Random Forest.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    
    @st.cache_data
    def load_template():
        preprocessed = pd.read_csv("preprocessed.csv")
        return preprocessed
    
    model = load_model()
    preprocessed = load_template()
    scaler = joblib.load("scaler.pkl")

    # Sidebar Inputs
    st.sidebar.header("Input Features")

    input_template = preprocessed.iloc[0:1]

    month = input_template['Month']= st.sidebar.selectbox("Select Month", [5, 6, 7, 8, 9, 10, 11])
    hour = input_template['Hour'] = st.sidebar.selectbox("Select Hour", [12, 13, 14, 15, 16, 17])
    location = st.sidebar.selectbox("Select Location", [
        "Puente Bilbao", "Puente Falbo", "Puente Irigoyen", "Arroyo Salguero", "Arroyo_Las Torres"
    ])
    
    # Replace values with user input
    pH = input_template['pH'] = st.sidebar.slider("pH Level", 0.0, 14.0, 7.0)
    conductivity = input_template['EC\n(µS/cm)'] = st.sidebar.slider("Conductivity (µS/cm)", 0.0, 5000.0, 1000.0)
    dissolved_oxygen = input_template['DO\n(mg/L)'] = st.sidebar.slider("Dissolved Oxygen (mg/L)", 0.0, 15.0, 7.0)
    turbidity = input_template['Turbidity (NTU)'] = st.sidebar.slider("Turbidity (NTU)", 0.0, 1000.0, 500.0)
    temp = input_template['Ambient temperature (°C)'] = st.sidebar.slider("Temperature (°C)", 0.0, 40.0, 25.0)
    level = input_template['Level (cm)'] = st.sidebar.slider("Level (cm)", 0.0, 100.0, 50.0)
    tss = input_template['TSS\n(mL sed/L)'] = st.sidebar.slider("TSS (mL sed/L)", 0.0, 700.0, 350.0)
    total_cl = input_template['Total Cl-\n(mg Cl-/L)'] = st.sidebar.slider("Total Cl- (mg Cl-/L)", 0.0, 200.0, 100.0)


    # One-hot encoding for location
    location_columns = [
        "Sampling point_Arroyo Salguero",
        "Sampling point_Arroyo_Las Torres",
        "Sampling point_Puente Bilbao",
        "Sampling point_Puente Falbo",
        "Sampling point_Puente Irigoyen"
    ]

    # Create empty one-hot dict
    location_encoding = {col: 0 for col in location_columns}

    # Map selected location to corresponding column name
    mapping = {
        "Puente Bilbao": "Sampling point_Puente Bilbao",
        "Puente Falbo": "Sampling point_Puente Falbo",
        "Puente Irigoyen": "Sampling point_Puente Irigoyen",
        "Arroyo Salguero": "Sampling point_Arroyo Salguero",
        "Arroyo_Las Torres": "Sampling point_Arroyo_Las Torres"
    }

    selected_column = mapping[location]
    location_encoding[selected_column] = 1

    # Combine numeric inputs with location one-hot columns
    input_data = pd.DataFrame([{
        "Month": month,
        "Hour": hour,
        "pH": pH,
        "Turbidity (NTU)": turbidity,
        "DO\n(mg/L)": dissolved_oxygen,
        "EC\n(µS/cm)": conductivity,
        "Ambient temperature (°C)": temp,
        "Level (cm)": level,
        "TSS\n(mL sed/L)": tss,
        "Total Cl-\n(mg Cl-/L)": total_cl,
        **location_encoding  # Spread the one-hot encoded location features
    }])

    

    # Predict button
    if st.button("Classify Water Quality"):

        # Ensure that the target column is removed from the input template
        if "Classification encoded" in input_template.columns:
            input_features = input_template.drop("Classification encoded", axis=1)
        else:
            input_features = input_template

        input_scaled = scaler.transform(input_features)
        prediction = model.predict(input_scaled)
        st.write("Raw prediction output:", prediction)

        prediction_int = int(prediction[0])
        
        mapping = {
            0: "Excellent (Support Aquatic life)", 
            1: "Good (Acceptable to some Aquatic life)", 
            2: "Poor (Pollution)", 
            3: "Very Poor (Hypoxic)"
        }
        
        result = mapping.get(prediction_int, "Unknown")

        st.success(f"Predicted Water Quality: {result}")

        st.write("### Input Summary")
        st.dataframe(input_features)

elif section == "About":
    st.info("This app predicts water quality based on parameters collected across different locations and times. The model used is a Random Forest classifier.")