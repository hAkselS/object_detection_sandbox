import streamlit as st
import time 

# Must be first ** Define the webpage 
st.set_page_config(
    page_title="MOUSS_mini with Fish No Fish",
    page_icon="🐠",
    layout="wide",
)

# Temp variables, try a class next 
drop_name = 'badingus'
drop_status = 'Ready'
cam1_image_count = -1
cam_2_image_count = -1
depth = -1
temp = -1
detect_yes_no = 'No'

# Web Variables 
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)

is_recording = False 

# Web Functions
# Function to display a red dot for recording status
def show_recording_status(placeholder, is_recording):
    if is_recording:
        placeholder.markdown("""
        <div style="display: flex; align-items: center;">
            <div style="height: 40px; width: 40px; background-color: red; border-radius: 50%; margin-right: 10px;"></div>
            <h3 style="color: red;">Recording</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        placeholder.markdown("<h3 style='color: green;'>Not Recording</h3>", unsafe_allow_html=True)

      

# Page title 
st.title("MOUSS_mini with Fish No Fish")

# Placeholder for variables
dropName = st.empty()
dropStatus = st.empty()
cam1Count = st.empty()
cam2Count = st.empty()
currentDepth = st.empty()
currentTemp = st.empty()
fishDetected = st.empty()

status_placeholder = st.empty()
image_placeholder = st.empty()
second_image_placeholder = st.empty()

# Here is where variables get updated every one second, on the website
for i in range(100):
    # Simulate variable changes (replace with real data updates in the future)
    cam1_image_count += 1
    cam_2_image_count += 1
    depth += 0.1
    temp += 0.2
    detect_yes_no = 'Yes' if i % 4 == 0 else 'No'

    # Update the placeholders
    dropName.markdown(f'<p class="big-font">Drop name: {drop_name}</p>', unsafe_allow_html=True)
    dropStatus.markdown(f'<p class="big-font">Drop status: {drop_status}</p>', unsafe_allow_html=True)
    cam1Count.markdown(f'<p class="big-font">Camera 1 image count: {cam1_image_count}</p>', unsafe_allow_html=True)
    cam2Count.markdown(f'<p class="big-font">Camera 2 image count: {cam_2_image_count}</p>', unsafe_allow_html=True)
    currentDepth.markdown(f'<p class="big-font">Current Depth (Meters): {depth:.1f}</p>', unsafe_allow_html=True)
    currentTemp.markdown(f'<p class="big-font">Current Temperature (Fahrenheit): {temp:.1f}</p>', unsafe_allow_html=True)
    fishDetected.markdown(f'<p class="big-font">Fish detected? {detect_yes_no}</p>', unsafe_allow_html=True)
    
    if i % 2 ==0:
        is_recording = True
    if i % 11 == 0:
        is_recording = False 
    if i > 10: 
        image_placeholder.image('test_code_2/rcnn_training/fish_data/fish_images/Screen Shot 2024-09-06 at 11.56.53 AM.png', caption='pretty good fish', width = 150)
    if i > 12: 
        second_image_placeholder.image('test_code_2/rcnn_training/fish_data/fish_images/Screen Shot 2024-09-06 at 12.17.48 PM.png', caption='second fish', width = 150)
    show_recording_status(status_placeholder, is_recording)
    time.sleep(1)
