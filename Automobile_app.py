import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np

def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.title("Automobile Price Prediction")

    model_options = ["Linear Regression", "Random Forest", "Decision Tree"]
    selected_model_name = st.selectbox("Select Model", model_options)

    model_dir = "models"  #adjust to your path.
    model_file_map = {
        "Linear Regression": "LinearRegression_model.joblib",
        "Random Forest": "RandomForestRegressor_model.joblib",
        "Decision Tree": "DecisionTreeRegressor_model.joblib",
    }

    model_path = os.path.join(model_dir, model_file_map[selected_model_name])
    model = load_model(model_path)

    if model is None:
        return

    st.subheader("Enter Car Features")

    # Input fields based on your DataFrame's columns
    symboling = st.selectbox("Symboling", [-2, -1, 0, 1, 2, 3])
    normalized_losses = st.number_input("Normalized Losses", value=122.0)
    fuel_type = st.selectbox("Fuel Type", [0, 1], format_func=lambda x: ['Diesel', 'Gas'][x])  # 0: diesel, 1: gas
    aspiration = st.selectbox("Aspiration", [0, 1], format_func=lambda x: ['Standard', 'Turbo'][x])  # 0: std, 1: turbo
    num_of_doors = st.selectbox("Number of Doors", [2,4])
    body_style = st.selectbox("Body Style", [0, 1, 2, 3, 4], format_func=lambda x: ['hatchback', 'wagon', 'sedan', 'hardtop', 'convertible'][x])
    drive_wheels = st.selectbox("Drive Wheels", [0, 1, 2], format_func=lambda x: ['4wd', 'fwd', 'rwd'][x])  # 0: 4wd, 1: fwd, 2: rwd
    engine_location = st.selectbox("Engine Location", [0, 1], format_func=lambda x: ['Front', 'Rear'][x])  # 0: front, 1: rear
    wheel_base = st.number_input("Wheel Base", value=98.8)
    length = st.number_input("Length", value=177.3)
    width = st.number_input("Width", value=66.5)
    height = st.number_input("Height", value=55.5)
    curb_weight = st.number_input("Curb Weight", value=2555)
    num_of_cylinders = st.selectbox("Number of Cylinders", [2, 3, 4, 5, 6, 8, 12])
    engine_size = st.number_input("Engine size", value = 70)
    engine_type = st.selectbox("Engine Type", [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ['ohcf', 'ohc', 'l', 'ohcv', 'rotor', 'dohc', 'dohcv'][x])
    bore = st.number_input("Bore", value=3.33)
    stroke = st.number_input("Stroke", value=3.25)
    compression_ratio = st.number_input("Compression Ratio", value=10.0)
    horsepower = st.number_input("Horsepower", value=103.0)
    peak_rpm = st.number_input("Peak RPM", value=5500.0)
    city_mpg = st.number_input("City MPG", value=24)
    highway_mpg = st.number_input("Highway MPG", value=30)
    fuel_system = st.selectbox("Fuel System", [0, 1, 2, 3, 4, 5, 6, 7], format_func=lambda x: ['mfi', 'spfi', '4bbl', 'spdi', '1bbl', 'idi', '2bbl', 'mpfi'][x])

    car_makes = {
        0: ['chevrolet', 'dodge', 'plymouth', 'honda', 'isuzu', 'mitsubishi', 'nissan', 'toyota', 'volkswagen'],
        1: ['mazda', 'subaru', 'renault', 'mercury', 'saab', 'peugot', 'alfa-romero'],
        2: ['audi', 'volvo', 'bmw', 'jaguar', 'mercedes-benz', 'porsche']
    }
    all_makes = [make for sublist in car_makes.values() for make in sublist]
    selected_make = st.selectbox("Car Maker",all_makes)
    selected_key = None
    for key, value in car_makes.items():
        if selected_make in value:
            selected_key = key
            break


    input_data = pd.DataFrame({
        "symboling": [symboling],
        "normalized-losses": [normalized_losses],
        "make": [selected_key],
        "fuel-type": [fuel_type],
        "aspiration": [aspiration],
        "num-of-doors": [num_of_doors],
        "body-style": [body_style],
        "drive-wheels": [drive_wheels],
        "engine-location": [engine_location],
        "wheel-base": [wheel_base],
        "length": [length],
        "width": [width],
        "height": [height],
        "curb-weight": [curb_weight],
        "engine-type": [engine_type],
        "num-of-cylinders": [num_of_cylinders],
        "engine-size": [engine_size],
        "fuel-system": [fuel_system],
        "bore": [bore],
        "stroke": [stroke],
        "compression-ratio": [compression_ratio],
        "horsepower": [horsepower],
        "peak-rpm": [peak_rpm],
        "city-mpg": [city_mpg],
        "highway-mpg": [highway_mpg]
    })

    if st.button("Predict Price"):
        try:
            prediction = model.predict(input_data)
            st.subheader("Predicted Price")
            if prediction > 1:
                st.write(f"${prediction[0]:.2f}")
            else:
                st.write(f"Please select the values properly for {model} to predict correct Price")

        except Exception as e:
            st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    main()