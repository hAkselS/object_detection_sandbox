#####################################################################
#####################################################################
##  File:   Dockerfile
##
##  Spec:   Dockerfile is used to create the ros-humble
##          docker image. This file gets the base image,
##          downloaded dependencies, and copys files into the 
##          image. This file also sets the entrypoint
##          and runs the startup.sh script at start up.
##
##  i/o:    None.
##
##  Author: Aksel Sloan | aksel.sloan@noaa.gov
##
##
#####################################################################
#####################################################################

#FROM --platform=linux/arm64 osrf/ros:humble-desktop
FROM arm64v8/ros:humble-ros-base-jammy

# Set the ROS distribution
ARG ROS_DISTRO=humble

# Update and install packages
RUN echo "hello world"
RUN apt-get update

RUN apt-get update && \
    apt-get install -y \
    vim \
    python3-pip \
    usbutils \
    python3-smbus \
    libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy workspace and scripts
COPY . /fish_no_fish
# Not used but there for backup and reference 
# COPY docker/requirements.txt /requirements.txt
# RUN pip3 install -r /requirements.txt

# Install Python packages
# TODO: add requirements.txt file 
# may need opencv-contrib-python
RUN python3 -m pip install \
    numpy \
    flask \ 
    opencv-python \
    datetime \
    pandas \
    ultralytics \
    streamlit \
    plotly \ 
    pillow 

CMD ["/bin/bash"]
