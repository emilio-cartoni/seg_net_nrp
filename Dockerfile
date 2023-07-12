FROM ubuntu:20.04
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install vim wget nano curl git git-lfs ca-certificates -y

RUN apt-get update
RUN apt-get install -y gnupg2

RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" | tee /etc/apt/sources.list.d/ros-focal.list
RUN curl http://repo.ros2.org/repos.key | apt-key add -
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -

RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install ros-noetic-desktop-full git-lfs ros-noetic-object-recognition-msgs ros-noetic-moveit-core ros-noetic-moveit-ros-perception ros-noetic-moveit-ros-planning-interface ros-noetic-velocity-controllers ros-noetic-twist-mux python3-rostopic ros-noetic-effort-controllers ros-noetic-position-controllers ros-noetic-joint-trajectory-controller ros-noetic-moveit ros-noetic-rviz-visual-tools ros-noetic-moveit-visual-tools -y

# Nano settings
RUN touch /root/.nanorc
RUN echo "set tabsize 4" >> ~/.nanorc
RUN echo "set tabstospaces" >> ~/.nanorc

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

ENV ROS_DISTRO=noetic
#FROM ros:noetic-perception
RUN apt-get update
RUN apt-get install -y git python3.8-venv

WORKDIR /workspace
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install wheel

# Install dependencies:
COPY requirements.txt .
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/$ROS_DISTRO/setup.bash && pip install -r requirements.txt
