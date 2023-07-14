import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from src.utils import onehot_to_rgb, hsv_to_rgb
import sys

dataset_dir = r'D:\DL\datasets\nrp'
dataset_type = 'multi_small'  # 'multi_small'; 'multi_shelf'
train_path = os.path.join(dataset_dir, dataset_type, 'train.hdf5')
valid_path = os.path.join(dataset_dir, dataset_type, 'valid.hdf5')
sample_dir = os.path.join(dataset_dir, dataset_type, 'camera_images')
segment_dir = os.path.join(dataset_dir, dataset_type, 'segment_images')
label_id_map = {'a_marbles': 0, 'b_marbles': 1, 'apple': 2, 'banana': 3, 'shelf': 4}
# label_id_map = {'chips_can': 0, 'hammer': 1, 'power_drill': 2,
#                 'scissors': 3, 'timer': 4, 'adjustable_wrench': 5,
#                 'flat_screwdriver': 6, 'phillips_screwdriver': 7}

n_sequences_train = 1600
n_sequences_valid = 400
n_frames_max = 59
h, w = 320, 320
n_channels = 3
n_classes = max(label_id_map.values()) + 1
no_hot_labels = False


show_string = input('Do you want to show the existing dataset?(y/N)')
if show_string.lower() == 'y':
    with h5py.File(train_path, 'r') as f:
        samples = np.array(f['samples'])
        labels = np.array(f['labels'])

    n_sequences_train, n_channels, h, w, n_frames_max = samples.shape

    print("Samples shape:", samples.shape)
    print("Labels shape:", labels.shape)

    if labels.shape[1] == 1:
        n_classes = labels.max()
    else:
        n_classes = labels.shape[1]

    for seq in range(n_sequences_train):
        print(f"Showing sequence {seq} of {n_sequences_train}")

        image_video = samples[seq, ...].transpose(1, 2, 0, 3)
        labels_seq = np.expand_dims(labels[seq,...], 0)

        # Fix in case labels are put all in the same dimension intead of having own dimensions
        labels_seq = np.hstack([labels_seq == (idx + 1)  for idx in range(n_classes)])

        labels_video = onehot_to_rgb(labels_seq)[0].transpose(1, 2, 0, 3)
        labels_video *= 255

        rows = 8  # use optimal tiling algorithm to calculate rows and cols
        cols = 7
        if rows*cols < n_frames_max:
            print("Part of the sequence will not be shown, insufficient rows or cols.")

        total_image = np.zeros((0, w * 2 * cols, 3))
        for r in range(rows):
            h_strip = np.zeros((h, 0, 3))
            for c in range(cols):
                frame_id = r*cols + c
                if frame_id < n_frames_max:
                    h_strip = np.hstack([h_strip, image_video[..., frame_id], labels_video[..., frame_id]])
                else:
                    h_strip = np.hstack([h_strip, np.zeros((h, w * 2, 3))])
            total_image = np.vstack([total_image, h_strip])
        plt.imshow(total_image.astype('uint8'))
        plt.show()
    sys.exit()
else:
    f_train = h5py.File(train_path, 'w')
    f_valid = h5py.File(valid_path, 'w')

img_train_array_shape = (n_sequences_train, n_channels, h, w, n_frames_max)
img_train_array = np.zeros(img_train_array_shape, dtype=np.uint8)
img_valid_array_shape = (n_sequences_valid, n_channels, h, w, n_frames_max)
img_valid_array = np.zeros(img_valid_array_shape, dtype=np.uint8)
for sequence_id, sequence_path in enumerate(sorted(os.listdir(sample_dir))):
    sequence_dir = os.path.join(sample_dir, sequence_path)
    if sequence_id < n_sequences_train + n_sequences_valid:
        print(f'\rDoing image sequence {sequence_id:04}', end='')
        for frame_id, img_path in enumerate(sorted(os.listdir(sequence_dir))):
            if frame_id < n_frames_max:
                img_full_path = os.path.join(sequence_dir, img_path)
                with Image.open(img_full_path) as read_img:
                    to_write = np.array(read_img).transpose((2, 0, 1))
                    if sequence_id < n_sequences_train:
                        img_train_array[sequence_id, :, :, :, frame_id] = to_write
                    else:
                        valid_id = sequence_id - n_sequences_train
                        img_valid_array[valid_id, :, :, :, frame_id] = to_write

f_train.create_dataset('samples', img_train_array_shape, data=img_train_array)
f_valid.create_dataset('samples', img_valid_array_shape, data=img_valid_array)
del img_train_array, img_valid_array

seg_train_array_shape = (n_sequences_train, 1, h, w, n_frames_max)
seg_train_array = np.zeros(seg_train_array_shape, dtype=np.uint8)
seg_valid_array_shape = (n_sequences_valid, 1, h, w, n_frames_max)
seg_valid_array = np.zeros(seg_valid_array_shape, dtype=np.uint8)
for sequence_id, segment_path in enumerate(sorted(os.listdir(segment_dir))):
    sequence_dir = os.path.join(segment_dir, segment_path)
    frame_id, frame_label = -1, -1
    if sequence_id < n_sequences_train + n_sequences_valid:
        print(f'\rDoing label sequence {sequence_id:04}', end='')
        for segment_path in sorted(os.listdir(sequence_dir)):
            new_frame_label = int(segment_path.split('seg_img_')[1].split('_')[1])
            if frame_label != new_frame_label:
                frame_id += 1
                frame_label = new_frame_label
            label_name = '_'.join(segment_path.split('__')[0].split('_')[4:])
            # print(label_name, label_id_map)
            if not (label_name[:label_name.find(":")]  in label_id_map.keys()):
                continue
            label_id = label_id_map[label_name[:label_name.find(":")]]

            if frame_id < n_frames_max:
                seg_full_path = os.path.join(sequence_dir, segment_path)
                with Image.open(seg_full_path) as read_seg:
                    to_write = np.array(read_seg)[:, :, -1]  # alpha-channel is seg
                    to_write = np.where(to_write > 0, 1, 0).astype(np.uint8)
                    to_write *= (label_id + 1)
                    if sequence_id < n_sequences_train:
                        seg_train_array[sequence_id, :, :, :, frame_id] += to_write
                    else:
                        valid_id = sequence_id - n_sequences_train
                        seg_valid_array[valid_id, :, :, :, frame_id] += to_write

if no_hot_labels:
    seg_train_array = np.argmax(seg_train_array, axis=1)
    seg_valid_array = np.argmax(seg_valid_array, axis=1)
    seg_train_array_shape = (n_sequences_train, h, w, n_frames_max)
    seg_valid_array_shape = (n_sequences_valid, h, w, n_frames_max)
f_train.create_dataset('labels', seg_train_array_shape, data=seg_train_array)
f_valid.create_dataset('labels', seg_valid_array_shape, data=seg_valid_array)
f_train.close()
f_valid.close()
