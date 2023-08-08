import cv2
import imageio
import numpy as np
from src.model import PredNet
import torchvision.transforms as transforms
from src.loss_fn import DiceLoss, FocalLoss
from src.load_inference import load_inference_model
import torch

id = np.random.randint(0, 2000)
image_path = f"./dataset_hbp/img/{id}/image_5.png"
label_path = f"./dataset_hbp/mask/{id}/image_5.npy"

import glob
ids = [a[24:28] for a in glob.glob('./hbp-data-3cameras/img/**/image_0.png')]
id = np.random.choice(ids)
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


device = 'cpu'
ckpt_path = './logs/Prednet_L-4_R-hgru_A-False_P-True_5728/version_0/checkpoints/epoch=9-step=1250.ckpt'
ckpt_path = './logs/Prednet_L-4_R-hgru_A-False_P-True_3375/version_0/checkpoints/epoch=19-step=2500.ckpt'

model, device = load_inference_model(ckpt_path, device='cpu', num_classes=2, rnn_type='hgru', axon_delay=False, pred_loss=True, n_layers=4)

im = tensor[None, :, :, :]
image = im.to(device=device)
for t in range(0, 3):
    print(t)
    _, _, seg_image = model(image, t)

#    number_of_channel = seg_image.shape[1]
#    for i in range(number_of_channel):
#        tmp = seg_image[0,i,:,:]
#        #tmp = tmp /tmp.max()
#        tmp = tmp * 255
#        tmp = tmp.detach().numpy()
#        tmp = tmp.astype(np.uint8)
#        cv2.imwrite("output/output_ch_" +str(i)+"_t_"+str(t)+".png",tmp)

#imageio.mimsave('output/finalIO.png', [(seg_image[0,:,:,:].detach().numpy().transpose(1,2,0)*255).astype('uint8')])


import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3)
axes[0, 0].imshow(label[0])
axes[0, 1].imshow(seg_image[0, 0, ...].detach().numpy())
axes[0, 2].imshow(image1)
axes[1, 0].imshow(label[1])
axes[1, 1].imshow(seg_image[0, 1, ...].detach().numpy())
axes[1, 2].imshow(image1)

for ax in axes.flatten():
    plt.colorbar(ax.get_children()[0], ax=ax)

plt.savefig(f'output/plot{id}_t{t}.png')
plt.show()


import sys
sys.exit(0)

loss = DiceLoss()
foc_loss_fn = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
ce_loss = torch.nn.CrossEntropyLoss()
bce_loss = torch.nn.BCELoss()
print("Focal loss random vs label:", foc_loss_fn.forward(transform(np.random.rand(480,640,2)), transform(label.transpose(1, 2, 0))))
print("Focal loss seg_image vs label:", foc_loss_fn.forward(seg_image[0], transform(label.transpose(1, 2, 0))))
#print("Focal loss seg_image vs replicated single channel:", foc_loss_fn.forward(seg_image[0], transform(img_label.transpose(1, 2, 0))))
#print("Focal loss seg_image vs single channel:", foc_loss_fn.forward(seg_image[0], transform(one_chan.transpose(1, 2, 0))))
print("Dice loss random vs label:", loss.forward(transform(np.random.rand(480,640,2))[None,:,:,:], transform(label.transpose(1, 2, 0))[None,:,:,:]))
print("Dice loss seg_image vs label:", loss.forward(seg_image[0][None,:,:,:], transform(label.transpose(1, 2, 0))[None,:,:,:]))
#print("Dice loss seg_image vs replicated single channel:", loss.forward(seg_image[0][None,:,:,:], transform(img_label.transpose(1, 2, 0))[None,:,:,:]))
#print("Dice loss seg_image vs single channel:",loss.forward(transform(seg_image2[0].transpose(1,2,0))[None,:,:,:], transform(one_chan.transpose(1, 2, 0))[None,:,:,:]))
print("CE loss random vs label:", ce_loss.forward(torch.special.logit(transform(np.random.rand(480,640,2))[None,:,:,:]), transform(label.transpose(1, 2, 0))[None,:,:,:]))
print("CE loss seg_image vs label:", ce_loss.forward(torch.special.logit(seg_image[0][None,:,:,:], eps=0.0001), transform(label.transpose(1, 2, 0))[None,:,:,:]))

m = torch.nn.Sigmoid()
#seg_image = m(torch.randn(seg_image.shape, requires_grad=True))
seg_image.retain_grad()
#Q = loss.forward(seg_image[0][None,:,:,:], transform(label.transpose(1, 2, 0))[None,:,:,:])
#Q = foc_loss_fn.forward(seg_image[0][None,:,:,:], transform(label.transpose(1, 2, 0))[None,:,:,:])
#Q = ce_loss(torch.special.logit(seg_image[0][None,:,:,:], eps=0.0001), transform(label.transpose(1, 2, 0))[None,:,:,:])
Q = bce_loss.forward(seg_image[0][None,:,:,:].double(), transform(label.transpose(1, 2, 0))[None,:,:,:].double())
Q.backward()

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3)
vmin = seg_image.grad.detach().numpy().min()
vmax = seg_image.grad.detach().numpy().max()
vmax = max(abs(vmin), abs(vmax))
vmin = min(-abs(vmin), -abs(vmax))
axes[0, 0].imshow(label[0])
axes[0, 1].imshow(seg_image[0, 0, ...].detach().numpy())
axes[0, 2].imshow(seg_image.grad[0, 0, ...].detach().numpy(), vmin=vmin, vmax=vmax)
axes[1, 0].imshow(label[1])
axes[1, 1].imshow(seg_image[0, 1, ...].detach().numpy())
axes[1, 2].imshow(seg_image.grad[0, 1, ...].detach().numpy(), vmin=vmin, vmax=vmax)

for ax in axes.flatten():
    plt.colorbar(ax.get_children()[0], ax=ax)


bb = seg_image.grad.detach().numpy()
axes[0, 2].set_title(f"{bb[0,0,...].min():.3E} {bb[0,0,...].max():.3E}")
axes[1, 2].set_title(f"{bb[0,1,...].min():.3E} {bb[0,1,...].max():.3E}")

plt.show()

