# Prednet Segmentation

## Setup

- Create a and load new venv: 
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```
- Install pytorch and additional packages:
  ```bash
  pip install wheel
  pip install -r requirements.txt
  ```

## Run setup

- Activate venv: `source venv/bin/activate`
- Run: `python test_ros.py`

## test_ros.py

The node subscribes to a camera image topic, here `/camera/camera/image`. It registers a series of image topics to output segmented images. An individual topic is used for each identified object, under `/prednet_segmentation/{item_name}/image`. In `test_ros.py`, we define three objects to identify. At this time, it's outputting noise and should only be used to setup the interface.