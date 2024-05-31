import pandas as pd
import streamlit as st

# Hyperparameters for each model
hyperparameters = {
    "Model": ["Model A", "Model B", "Model C"],
    "Learning Rate": [0.001, 0.0005, 0.0007],
    "Batch Size": [32, 64, 32],
    "Epochs": [10, 12, 15],
    "Dropout": [0.1, 0.2, 0.15]
}

# Convert to DataFrame
df_hyperparameters = pd.DataFrame(hyperparameters)

st.markdown("# Biomedical Shared Task of WMT 24 - Hyperparameters")
st.dataframe(df_hyperparameters, width=1000)
