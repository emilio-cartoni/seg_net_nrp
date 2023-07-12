FROM ros:noetic-perception
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
