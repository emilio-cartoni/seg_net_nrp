import torch
import numpy as np
import imageio
from src.model import PredNet
import cv2

n_classes = 1

def load_image_sequence():
    list_of_torch_tensors = []
    for i in range(20):
        list_of_torch_tensors.append(torch.rand(1, 3, 256, 256))
    return list_of_torch_tensors  # of shape (1, n_channels, width, height)

def plot_seg_image_sequence(seg_image_sequence):
    # your code here
    n_frames = len(seg_image_sequence)
    seg_image_stack = np.stack(seg_image_sequence, axis=-1)
    seg_image_rgb = onehot_to_rgb(seg_image_stack)
    seg_image_list = [seg_image_rgb[..., t] for t in range(n_frames)]
    imageio.mimsave('./seg_output.gif', seg_image_list, duration=0.1)

def onehot_to_rgb(onehot_array):
    batch_size, num_classes, w, h, n_frames = onehot_array.shape
    rgb_tensor = np.zeros((batch_size, 3, w, h, n_frames))
    hue_space = np.linspace(0.0, 1.0, num_classes + 1)[:-1]
    rgb_space = [hsv_to_rgb(hue) for hue in hue_space]
    for n in range(num_classes):
        class_tensor = onehot_array[:, n]
        for c, color in enumerate(rgb_space[n]):
            rgb_tensor[:, c] += color * class_tensor
    return rgb_tensor

def hsv_to_rgb(hue):
    v = 1 - abs((int(hue * 360) / 60) % 2 - 1)
    hsv_space = [
        [1, v, 0], [v, 1, 0], [0, 1, v],
        [0, v, 1], [v, 0, 1], [1, 0, v]]
    return hsv_space[int(hue * 6)]

if __name__ == '__main__':
    device = 'cuda'  # or 'cpu'
    model = PredNet(model_name='my_model',
                    n_classes=n_classes,
                    n_layers=3,
                    seg_layers=(1, 2),
                    bu_channels=(64, 128, 256),
                    td_channels=(64, 128, 256),
                    do_segmentation=True,
                    device=device)
    model.eval()
    seg_image_sequence = []
    image_sequence = load_image_sequence()
    #n_frames = image_sequence.shape[-1]
    with torch.no_grad():
        t = 0
        for im in image_sequence:
            #image = image.to(device=device)
            image = im.to(device=device)
            _, _, seg_image = model(image, t)
            seg_image_numpy = seg_image.cpu().numpy()   # Shape of (1, n_channels, width_height)
            seg_image_sequence.append(seg_image_numpy)
            t += 1
    
    #plot_seg_image_sequence(seg_image_sequence)
