import torch
import numpy as np
import random
from src.model import PredNetVGG
from src.utils import train_fn, valid_fn
from src.dataset_fn import get_datasets_seg, get_SQM_dataset
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Parameters
load_model, n_epochs_run, n_epoch_save, epoch_to_load = False, 1000, 5, None
name = 'Alban'
do_time_aligned, do_train_vgg, do_untouched_bu = True, False, False
batch_size_train, batch_size_valid = 1, 4
vgg_type, n_layers, t_start = 'vgg19', 5, 10
pr_layers = tuple([l for l in [0, 1, 2, 3, 4] if l < n_layers])  # set as [] for not doing it
sg_layers = tuple([l for l in [0, 1, 2, 3, 4] if l < n_layers])  # set as [] for not doing it
td_channels = td_channels = (64, 128, 256, 512, 512)
td_channels = td_channels[:n_layers]
learning_rate, dropout_rates = 1e-3, (0.0, 0.0, 0.0, 0.0, 0.0)[:n_layers]
lr_decay_time, lr_decay_rate = 1, 0.9
loss_w = {
    'latent': (1.0, 1.0, 1.0, 1.0, 1.0)[:n_layers],
    'img_bce': 0.0 if len(pr_layers) > 0 else 0.0,
    'img_mae': 1.0 if len(pr_layers) > 0 else 0.0,
    'img_mse': 0.0 if len(pr_layers) > 0 else 0.0,
    'seg_bce': 10.0 if len(sg_layers) > 0 else 0.0,
    'seg_mse': 0.0 if len(sg_layers) > 0 else 0.0,
    'seg_foc': 0.0 if len(sg_layers) > 0 else 0.0,
    'seg_dice': 0.0 if len(sg_layers) > 0 else 0.0}
model_name = \
      f'{vgg_type}_TA{int(do_time_aligned)}_BU{int(do_untouched_bu)}'\
    + f'_TD{td_channels}_PR{pr_layers}_SG{sg_layers}'\
    + f'_DR{tuple([int(10 * r) for r in dropout_rates])}'\
    + f'_{name}'
model_name = model_name.replace('.', '-').replace(',', '-').replace(' ', '').replace("'", '')
do_prediction = not (len(pr_layers) == 0 or sum([loss_w['img_' + k] for k in ['bce', 'mae', 'mse']]) == 0)
do_segmentation = not (len(sg_layers) == 0 or sum([loss_w['seg_' + k] for k in ['bce', 'mse', 'foc', 'dice']]) == 0)

# Dataset
# dataset_path = r'C:\Users\loennqvi\Github\seg_net_vgg\data\SQM'
dataset_path = r'D:\DL\datasets\kitti\mots'
n_samples, tr_ratio = 1000, 0.80  # n_train(valid)_samples = ((1-)tr_ratio) * n_samples
n_frames, n_backprop_frames = 100, 1
augmentation, remove_ground = True, False
n_classes = 3 if remove_ground else 4
train_dl, valid_dl = get_datasets_seg(
    dataset_path, tr_ratio, batch_size_train, batch_size_valid, n_frames,
    augmentation=augmentation, n_classes=n_classes, remove_ground=remove_ground)
# sqm_dataset_path = r'D:\DL\datasets\kitti\mots\sqm'
# valid_dl = get_SQM_dataset(sqm_dataset_path, n_classes, remove_ground)

# Load the model
if not load_model:
    print(f'\nCreating model: {model_name}')
    model = PredNetVGG(
        model_name, vgg_type, n_classes, n_layers, pr_layers, sg_layers,
        td_channels, dropout_rates, do_time_aligned, do_untouched_bu,
        do_train_vgg, do_prediction, do_segmentation)
    train_losses, valid_losses, last_epoch = [], [], 0
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        range(lr_decay_time, (n_epochs_run + 1) * 10, lr_decay_time),
        gamma=lr_decay_rate)
    train_losses, valid_losses = [], []
else:
    print(f'\nLoading model: {model_name}')
    model, optimizer, scheduler, train_losses, valid_losses = \
        PredNetVGG.load_model(model_name, epoch_to_load)
    last_epoch = scheduler.last_epoch

# Train the network
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Training network ({n_params} trainable parameters)')
for epoch in range(last_epoch, last_epoch + n_epochs_run):
    print(f'\nEpoch n°{epoch}')
    train_losses.append(train_fn(
        train_dl, model, optimizer, loss_w, t_start, n_backprop_frames, epoch))
    valid_losses.append(valid_fn(valid_dl, model, loss_w, t_start, epoch))
    scheduler.step()
    if (epoch + 1) % n_epoch_save == 0:
        model.save_model(optimizer, scheduler, train_losses, valid_losses)
