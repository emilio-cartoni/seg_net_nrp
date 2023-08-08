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

max_t = 3
fig, axes = plt.subplots(3, max_t + 1)
axes[0, 0].imshow(label[0], vmin=0, vmax=1)
axes[1, 0].imshow(label[1], vmin=0, vmax=1)
axes[2, 0].imshow(image1)

for t in range(0, max_t):
    print(t)
    _, pred_image, seg_image = model(image, t)

    axes[0, t + 1].imshow(seg_image[0, 0, ...].detach().numpy())
    axes[1, t + 1].imshow(seg_image[0, 1, ...].detach().numpy())
    axes[2, t + 1].imshow(pred_image[0, ...].detach().numpy().transpose(1, 2, 0))


Path(f"output/{network_seed}").mkdir(parents=True, exist_ok=True)
fig.tight_layout()
plt.savefig(f'output/{network_seed}/plot{id}_t{t}.png')
