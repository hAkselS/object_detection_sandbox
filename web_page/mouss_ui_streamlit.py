import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development

import os 

# TODO: Convert to class 

cwd = os.getcwd()
print(cwd)

dataset_path = os.path.join(cwd, 'detections.csv')

# Page configuration 
st.set_page_config(
    page_title="MOUSS_mini with Fish No Fish",
    page_icon="✅",
    layout="wide",
)

@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv(dataset_path)

df = get_data()



# Visible configurations
st.title("MOUSS_mini with Fish No Fish")

# Top-level filters
job_filter = st.selectbox("Select an Image", pd.unique(df["Image"]))

