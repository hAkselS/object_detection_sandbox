import os
import time  # to simulate a real time data, time loop
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development


# Reference: 
# - Main source
# - https://blog.streamlit.io/how-to-build-a-real-time-live-dashboard-with-streamlit/
# - How to run streamlit from inside a python file
# - https://stackoverflow.com/questions/62760929/how-can-i-run-a-streamlit-app-from-within-a-python-script



# Set Streamlit page configuration first
st.set_page_config(
    page_title="MOUSS_mini with Fish No Fish",
    page_icon="✅",
    layout="wide",
)

# TODO: Convert to class

cwd = os.getcwd()
print(cwd)

dataset_path = os.path.join(cwd, 'detections.csv')

@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv(dataset_path)

df = get_data()

# Visible configurations
st.title("MOUSS_mini with Fish No Fish")

# Top-level filters
image_filter = st.selectbox("Select the Image", pd.unique(df["Image"]))

# Dataframe filter
df = df[df['Image'] == image_filter]

# Create three columns
kpi1, kpi2, kpi3 = st.columns(3)

# Fill in those three columns with respective metrics or KPIs
kpi1.metric(
    label="Detections 🍑",
    value=round(df["Num_detections"]),
)

# TODO: Doesn't work
# Find the average of the point before and after the point
# Calculate local average
# Ensure that there are at least 3 rows for this calculation
if len(df) >= 3:
    # Extract the last detection, the first detection, and the second-to-last detection
    try:
        current_point = df["Num_detections"].iloc[-1]  # Last point
        previous_point = df["Num_detections"].iloc[-2]  # One before last
        first_point = df["Num_detections"].iloc[0]      # First point
        # Calculate the local average of the previous, current, and first point
        local_avg = np.mean([previous_point, current_point, first_point])
    except IndexError as e:
        st.error(f"Indexing error while calculating local average: {e}")
        local_avg = None
else:
    st.warning("Not enough data to calculate local average (need at least 3 detections).")
    local_avg = None

# Fill in the second column with the local average (if it was computed)
kpi2.metric(
    label="Local Average 💋",
    value=local_avg if local_avg is not None else "N/A"
)

st.markdown("### Detailed Data View")
st.dataframe(df)

# creating a single-element container.
placeholder = st.empty()

for seconds in range(200):
    df["New_Num_detections"] = df["Num_detections"] * np.random.choice(range(1, 5))

    with placeholder.container():
        
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(
            label="Detections 🍑",
            value=round(df["Num_detections"]),
        )

        st.markdown("### Detailed Data View")
        st.dataframe(df)
    time.sleep(1)


