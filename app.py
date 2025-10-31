
import streamlit as st
import pandas as pd
import numpy as np
import joblib

#  Page Configuration 
st.set_page_config(
    page_title="Mobile Price Predictor",
    layout="centered"
)


@st.cache_resource
def load_assets():
    """Loads the trained model and scaler from disk."""
    try:
        model = joblib.load("mobile_price_model.pkl")
        scaler = joblib.load("mobile_price_scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Make sure 'mobile_price_model.pkl' and 'mobile_price_scaler.pkl' are in the same directory.")
        return None, None

model, scaler = load_assets()

# model was trained on
MODEL_COLUMNS = ['battery_power', 'refresh_rate', 'dual_sim', 'fc', 'five_g', 'int_memory',
                 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'ram', 'sc_h', 'sc_w',
                 'four_g', 'screen_megapixels']

# Preprocessing Function 

def preprocess_data_updated(df):
    df_processed = df.copy()
    
    df_processed['screen_megapixels'] = (df_processed['px_height'] * df_processed['px_width']) / 1_000_000
    
    if 'ram' in df_processed.columns:
        df_processed['ram'] = (df_processed['ram'] / 1024).round(2)
        
    if 'clock_speed' in df_processed.columns:
        df_processed.rename(columns={'clock_speed': 'refresh_rate'}, inplace=True)
        df_processed['refresh_rate'] = 60 + (df_processed['refresh_rate'] * 15)

    if 'four_g' in df_processed.columns and 'three_g' in df_processed.columns:
        df_processed.rename(columns={'four_g': 'five_g', 'three_g': 'four_g'}, inplace=True)
        df_processed.loc[df_processed['five_g'] == 1, 'four_g'] = 1
        
    df_processed.drop(columns=['px_height', 'px_width'], inplace=True, errors='ignore')
    
    return df_processed

# User Interface 
st.title("Mobile Price Range Predictor")
st.markdown("Use the sidebar to enter the phone's specifications and click 'Predict' to see the result.")

st.sidebar.header("Phone Features")

def user_input_features():
    """Creates sidebar widgets and returns user inputs as a DataFrame."""
    battery_power = st.sidebar.slider('Battery Power (mAh)', 500, 2000, 1200)
    clock_speed = st.sidebar.slider('Clock Speed (GHz)', 0.5, 3.0, 1.5, 0.1)
    dual_sim = st.sidebar.selectbox('Dual SIM', ('No', 'Yes'))
    
  
    connectivity = st.sidebar.radio(
        'Mobile Technology',
        ('5G', '4G'),
        index=1 
    )

    fc = st.sidebar.slider('Front Camera (MP)', 0, 20, 5)
    int_memory = st.sidebar.slider('Internal Memory (GB)', 2, 64, 32)
    m_dep = st.sidebar.slider('Mobile Depth (cm)', 0.1, 1.0, 0.5, 0.1)
    mobile_wt = st.sidebar.slider('Mobile Weight (g)', 80, 200, 140)
    n_cores = st.sidebar.slider('Number of Cores', 1, 8, 4)
    pc = st.sidebar.slider('Primary Camera (MP)', 0, 21, 10)
    px_height = st.sidebar.slider('Pixel Resolution Height', 0, 1960, 640)
    px_width = st.sidebar.slider('Pixel Resolution Width', 0, 1988, 1280)
    ram = st.sidebar.slider('RAM (MB)', 256, 4096, 2048)
    sc_h = st.sidebar.slider('Screen Height (cm)', 5, 19, 12)
    sc_w = st.sidebar.slider('Screen Width (cm)', 0, 18, 5)

    map_yes_no = {'No': 0, 'Yes': 1}

    

    if connectivity == '5G':
        original_four_g = 1  
        original_three_g = 1  #
    else:  
        original_four_g = 0
        original_three_g = 1

    data = {
        'battery_power': battery_power, 'clock_speed': clock_speed, 'dual_sim': map_yes_no[dual_sim],
        'fc': fc, 'four_g': original_four_g, 'three_g': original_three_g,
        'int_memory': int_memory, 'm_dep': m_dep, 'mobile_wt': mobile_wt, 'n_cores': n_cores, 'pc': pc,
        'px_height': px_height, 'px_width': px_width, 'ram': ram, 'sc_h': sc_h, 'sc_w': sc_w
    }
    
    return pd.DataFrame(data, index=[0])

# Main Application Logic 
if model and scaler:
    input_df = user_input_features()

    if st.button('Predict Price Range'):
        # Preprocess the user input
        processed_input = preprocess_data_updated(input_df)
        
 
        processed_input = processed_input.reindex(columns=MODEL_COLUMNS, fill_value=0)
        
      
        scaled_input = scaler.transform(processed_input)

       
        prediction = model.predict(scaled_input)
        
     
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(scaled_input)
        else:
            prediction_proba = None

      
        price_range_map = {0: 'Low Cost', 1: 'Medium Cost', 2: 'High Cost', 3: 'Very High Cost'}
        predicted_label = price_range_map.get(prediction[0], "Unknown")

        st.subheader('Prediction Result')
        st.success(f'The predicted price range for this phone is: **{predicted_label}**')

        if prediction_proba is not None:
            st.subheader('Prediction Probabilities')
            proba_df = pd.DataFrame(prediction_proba, columns=price_range_map.values())
            st.write(proba_df)
            