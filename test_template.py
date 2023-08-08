import cv2
import imageio
import numpy as np
from src.model import PredNet
import torchvision.transforms as transforms
from src.loss_fn import DiceLoss, FocalLoss
from src.load_inference import load_inference_model
import torch
from pathlib import Path
import glob
import matplotlib.pyplot as plt


id = np.random.randint(0, 2000)
image_path = f"./dataset_hbp/img/{id}/image_5.png"
label_path = f"./dataset_hbp/mask/{id}/image_5.npy"

ids = [a[24:28] for a in glob.glob('./hbp-data-3cameras/img/**/image_0.png')]
id = SOMENUMBER
image_path = f"./hbp-data-3cameras/img/{id}/image_0.png"
label_path = f"./hbp-data-3cameras/mask/{id}/mask_0.npy"


label = np.load(label_path, allow_pickle=True)
label = label.transpose([2,0,1])
label = label * 255
print(label.shape)


image = cv2.imread(image_path)
image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
transform = transforms.ToTensor()
tensor = transform(image1)

network_seed = SOMENET
net_dir = glob.glob(f'./logs/Prednet*{network_seed}')[0]
ckpt_path = glob.glob(net_dir+'/version_0/checkpoints/*.ckpt')[0]
model, device = load_inference_model(ckpt_path, device='cpu', num_classes=2, rnn_type='hgru', axon_delay=False, pred_loss=True, n_layers=4)

im = tensor[None, :, :, :]
image = im.to(device=device)

cm = plt.get_cmap('viridis')
h_spacer = np.zeros((1, image1.shape[1], 3))
v_spacer = np.zeros((image1.shape[0] * 3 + 2, 1, 3))
png_image = cm(label[0])[..., :3] * 255
png_image = np.vstack([png_image, h_spacer, cm(label[1])[..., :3] * 255])
png_image = np.vstack([png_image, h_spacer, image1])

max_t = 3
for t in range(0, max_t):
    print(t)
    _, pred_image, seg_image = model(image, t)
    t_image = cm(seg_image[0, 0, ...].detach().numpy())[..., :3] * 255
    t_image = np.vstack([t_image, h_spacer, cm(seg_image[0, 1, ...].detach().numpy())[..., :3] * 255])
    t_image = np.vstack([t_image, h_spacer, pred_image[0].detach().numpy().transpose(1, 2, 0) * 255])
    png_image = np.hstack([png_image, v_spacer, t_image])

Path(f"output/{network_seed}").mkdir(parents=True, exist_ok=True)
imageio.mimsave(f'output/{network_seed}/plot{id}_t{t}.png', [png_image.astype('uint8')])
