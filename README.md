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
- Run: `python ros_img_segmentation.py`

## ros\_img\_segmentation.py

The node subscribes to a camera image topic, here `/camera/camera/image`. It registers a series of image topics to output segmented images. An individual topic is used for each identified object, under `/prednet_segmentation/{item_name}/image`. In `test_ros.py`, we define three objects to identify. At this time, it's outputting noise and should only be used to setup the interface.

## Segmentable meshes

At the moment, the model can segment the following meshes from the YCB dataset:

- plate
- sugar\_box
- tomato\_soup\_can

`ros_img_segmentation.py` outputs the individual segmented images via their respective topics.

## Train a new model

Requirements:

- A Folder called `multi_shelf` containing `camera_images` and `segment_images`
- Each folder should contain an individual subfolder for each image sequence, labeled from `0` to `number of sequences`
- Per sequence, the `camera_images` subfolders  should contain images labeled with the sequence id and the timestamp at which the image was taken as `cam_img_<sequence_id>_<timestamp>.png` (ex. `cam_img_00012_00001644.png`)
- Per sequence, the `segment_images` subfolders should contain ground truth images labeled with the sequence id, timestamp, and the segmented object name

Setup:

- Move `multi_shelf` folder into `dataset`
- In `generate_h5.py`, adjust line 7 to point to `dataset` folder
- In `train_net.py`, adjust line 32 to point to `multi_shelf` folder
- Run `generate_h5.py` to convert sequences to .h5 file
- Run `train_net.py` to train network (new network will be generated in `ckpt` folder)
- To use new network, adjust `model_name` in `ros_img_segmentation.py`, line 143 with new model name

