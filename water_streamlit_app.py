import streamlit as st
import pandas as pd
import joblib
import pickle

st.markdown(
    """
    <style>
    /* Set background for the whole app */
    [data-testid="stAppViewContainer"] {
        background-image: url('https://cff2.earth.com/uploads/2023/09/12105550/River-water-quality-1400x850.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Set background for the sidebar */
    [data-testid="stSidebar"] > div:first-child {
        background-image: url('data:image/jpeg;base64,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');
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