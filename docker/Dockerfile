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


# Install dependencies:
WORKDIR /root
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN /root/Miniconda3-py38_4.10.3-Linux-x86_64.sh -b && eval "$(/root/miniconda3/bin/conda shell.bash hook)" && /root/miniconda3/bin/conda clean -afy
RUN /root/miniconda3/bin/conda init
RUN echo 'conda activate prednet' >> ~/.bashrc
COPY environment.yml .
RUN /root/miniconda3/bin/conda env create -f /root/environment.yml && /root/miniconda3/bin/conda clean -afy

# Custom fix on keras saving.py
RUN sed -i  "s/.encode('utf8')//g" /root/miniconda3/envs/prednet/lib/python3.6/site-packages/keras/engine/saving.py
RUN sed -i  "s/.encode('utf-8')//g" /root/miniconda3/envs/prednet/lib/python3.6/site-packages/keras/engine/saving.py
RUN sed -i  "s/.decode('utf8')//g" /root/miniconda3/envs/prednet/lib/python3.6/site-packages/keras/engine/saving.py
RUN sed -i  "s/.decode('utf-8')//g" /root/miniconda3/envs/prednet/lib/python3.6/site-packages/keras/engine/saving.py

WORKDIR /workspace
