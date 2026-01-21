import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np

# Define the model architecture
class BreastCancerModel(nn.Module):
    def __init__(self):
        super(BreastCancerModel, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = BreastCancerModel()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    return model, scaler

# Load model and scaler
model, scaler = load_model_and_scaler()

# Streamlit UI
st.title("🏥 Breast Cancer Prediction")
st.markdown("Enter patient measurements to predict whether the tumor is **Benign** or **Malignant**")

col1, col2 = st.columns(2)

with col1:
    radius = st.number_input("Radius", min_value=0.0, value=10.0, step=0.1)

with col2:
    texture = st.number_input("Texture", min_value=0.0, value=10.0, step=0.1)

if st.button("🔍 Predict", use_container_width=True):
    # Normalize input
    input_data = np.array([[radius, texture]])
    input_normalized = scaler.transform(input_data)
    
    # Make prediction
    input_tensor = torch.FloatTensor(input_normalized)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    # Display result
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction > 0.5:
            st.error("🔴 **MALIGNANT**")
            st.write(f"Confidence: {prediction*100:.2f}%")
        else:
            st.success("🟢 **BENIGN**")
            st.write(f"Confidence: {(1-prediction)*100:.2f}%")
    
    with col2:
        st.metric("Prediction Score", f"{prediction:.4f}")

