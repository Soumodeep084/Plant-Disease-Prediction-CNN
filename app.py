import streamlit as st
import pandas as pd
from utils.preprocessing_utils import predict_disease_from_model 

# Load the dataset
disease_df = pd.read_csv("info/disease_info.csv", encoding="latin-1")

def get_disease_info(idx):
    row = disease_df[disease_df['index'] == idx]
    if not row.empty:
        return row.iloc[0]['description']
    else:
        return "No information available."

def get_disease_possible_steps(idx):
    row = disease_df[disease_df['index'] == idx]
    if not row.empty:
        return row.iloc[0]['Possible Steps']
    else:
        return "No information available."

st.markdown(
    """
     <style>
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }
        .css-z5fcl4 {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }
        header {visibility: hidden;}
        .css-1vq4p4l {padding: 0rem;}
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 Plant Disease Detection System")
st.text("This Engine will Help you to Detect Disease for Plants" , )

# Session state to handle file reset
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# File uploader
uploaded_file = st.file_uploader("Upload an image of Leaf of the Plant", type=["jpg", "png", "jpeg"])

# Save file to session state
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file

if st.session_state.uploaded_file is not None:
    st.image(st.session_state.uploaded_file, caption="Uploaded Image", width=250)

    if st.button("Predict Disease"):        
        
        pred = predict_disease_from_model(st.session_state.uploaded_file)

        disease_name = disease_df.iloc[pred , 1]
        disease_info = get_disease_info(pred)
        disease_possible_steps = get_disease_possible_steps(pred)

        st.success(f"Predicted Disease: **{disease_name}**")

        # Create two columns
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Disease Description :**\n{disease_info}")
        with col2:
            st.info(f"**Possible Prevention Steps :**\n{disease_possible_steps}")
            

# Test Again button to clear the uploaded file
if st.session_state.uploaded_file and st.button("Test Another Image"):
    st.session_state.uploaded_file = None
    st.rerun()  # Rerun the app to reset uploader
    

st.markdown("""
    <hr>
    <p style='text-align: center;'>
        Created by <b>Soumodeep Dutta</b> © 2025
    </p>
""", unsafe_allow_html=True)
