import torch
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from src.model import PredNet
from src.utils import train_fn, valid_fn
from src.dataset_fn_nrp import get_nrp_dataloaders
from src.dataset_fn_mots import get_mots_dataloaders, get_sqm_dataloaders
from src.dataset_fn_handover import get_handover_dataloaders
# from src.dataset_fn_bmw import get_bmw_dataloaders
# from src.dataset_fn_bmw2 import get_bmw_dataloaders
from src.dataset_fn_bmw3 import get_bmw_dataloaders
# from src.dataset_fn_multi import get_multi_dataloaders
from src.dataset_fn_multi2 import get_multi_dataloaders

# Model parameters
load_model, do_time_aligned, n_layers = False, True, 3
do_untouched_bu = False
batch_size_train, batch_size_valid = 4, 16
prd_layers = tuple([l for l in [0] if l < n_layers])
seg_layers = tuple([l for l in [1, 2] if l < n_layers])
bu_channels = (64, 128, 256, 512)[:n_layers]
td_channels = (64, 128, 256, 512)[:n_layers]
td_layers = ('H', 'H', 'H', 'H')[:n_layers]  # 'Hgru', 'Illusory', 'Lstm', 'Conv'
dropout_rates = (0.0, 0.0, 0.0, 0.0)[:n_layers]
device = 'cpu'  # 'cuda', 'cpu'

# Training parameters
n_epochs_run, n_epochs_save, epoch_to_load = 1000, 5, None
learning_rate, lr_decay_time, lr_decay_rate, betas = 1e-4, 50, 0.75, (0.9, 0.98)
first_cycle_steps, cycle_mult, max_lr, min_lr, warmup_steps, gamma = 10, 1.0, 1e-4, 1e-5, 2, 1.0
scheduler_type = 'multistep'  # 'multistep', 'cosannealwarmuprestart'
loss_w = {
    'latent': (5.0, 0.0, 0.0, 0.0)[:n_layers],  # now this is image loss
    'prd_mae': 3.0 if len(prd_layers) > 0 else 0.0,  # put to zero?
    'prd_mse': 0.0 if len(prd_layers) > 0 else 0.0,
    'prd_bce': 0.0 if len(prd_layers) > 0 else 0.0,
    'seg_mae': 0.0 if len(seg_layers) > 0 else 0.0,
    'seg_mse': 0.0 if len(seg_layers) > 0 else 0.0,
    'seg_bce': 0.0 if len(seg_layers) > 0 else 0.0,
    'seg_foc': 1.0 if len(seg_layers) > 0 else 0.0,
    'seg_dic': 1.0 if len(seg_layers) > 0 else 0.0}
do_prediction = not (len(prd_layers) == 0 or \
    sum([loss_w['prd_' + k] for k in ['mae', 'mse', 'bce']]) == 0)
do_segmentation = not (len(seg_layers) == 0 or \
    sum([loss_w['seg_' + k] for k in ['mae', 'mse', 'bce', 'foc', 'dic']]) == 0)

# Build model name
model_name = \
      f'TA{int(do_time_aligned)}_BU{bu_channels}_TD{td_channels}_TL{td_layers}'\
      f'_PL{prd_layers}_SL{seg_layers}_DR{tuple([int(10 * r) for r in dropout_rates])}'
model_name = model_name.replace('.', '-').replace(',', '-').replace(' ', '').replace("'", '')

# Dataset
dataset_type = 'multi_objects_background'  # 'mots', 'mots_sqm', 'nrp', 'handover', 'multi_objects', 'multi_objects_background', 'bmw', 'ovis'
n_frames, n_backprop_frames, t_start = 55, 5, n_layers
augmentation, remove_ground, tr_ratio = True, True, 0.8

if dataset_type == 'nrp':
    # dataset_path = r'D:\DL\datasets\nrp\training_room\training_room_dataset_00.h5'  # 1000 x 128 x 128 x 3
    dataset_path = r'D:\DL\datasets\nrp\training_room\training_room_dataset_01.h5'  # 300 x 320 x 320 x 3
    n_samples, n_classes = 300, 3 + int(not remove_ground)
    train_dl, valid_dl = get_nrp_dataloaders(dataset_path, tr_ratio, n_samples, 
                                             batch_size_train, batch_size_valid,
                                             n_classes, augmentation=augmentation,
                                             remove_ground=remove_ground, speedup_factor=1)

elif 'mots' in dataset_type:
    dataset_path = r'D:\DL\datasets\kitti\mots'
    sqm_dataset_path = r'D:\DL\datasets\kitti\mots\sqm'
    n_classes = 3 + int(not remove_ground)
    train_dl, valid_dl = get_mots_dataloaders(dataset_path, tr_ratio, batch_size_train,
                                              batch_size_valid, n_frames, augmentation=augmentation,
                                              n_classes=n_classes, remove_ground=remove_ground)
    if 'sqm' in dataset_type:
        sqm_losses = []
        sqm_dl = get_sqm_dataloaders(sqm_dataset_path, n_classes, remove_ground)

elif dataset_type in ['handover', 'multi_objects', 'multi_objects_background', 'bmw']:
    dataset_path = r'D:\DL\datasets\nrp' + fr'\{dataset_type}'
    dataloader_fn_dict = {'handover': get_handover_dataloaders,
                          'multi_objects': get_multi_dataloaders,
                          'multi_objects_background': get_multi_dataloaders,
                          'bmw': get_bmw_dataloaders}
    dataloader_fn = dataloader_fn_dict[dataset_type]
    train_dl, valid_dl, n_classes = dataloader_fn(dataset_path, tr_ratio,
                                                  batch_size_train, batch_size_valid, n_frames,
                                                  augmentation=augmentation, remove_ground=remove_ground)

# elif dataset_type == 'ovis':
#     dataset_path = r'D:\DL\datasets\ovis'
#     train_dl, valid_dl, n_classes = get_ovis_dataloaders(dataset_path, tr_ratio,
#                                                          batch_size_train, batch_size_valid, n_frames,
#                                                          augmentation=augmentation, remove_ground=remove_ground)

# Load the model
if not load_model:
    print(f'\nCreating model: {model_name}')
    model = PredNet(model_name,
                    n_classes,
                    n_layers,
                    seg_layers,
                    bu_channels,
                    td_channels,
                    do_segmentation,
                    device)
    train_losses, valid_losses, last_epoch = [], [], 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=betas)
    if scheduler_type == 'multistep':
        milestones=range(lr_decay_time,10 * (n_epochs_run + 1), lr_decay_time)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones,
                                                         lr_decay_rate)
    elif scheduler_type == 'cosannealwarmuprestart':
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                  first_cycle_steps=first_cycle_steps,
                                                  cycle_mult=cycle_mult,
                                                  max_lr=max_lr,
                                                  min_lr=min_lr,
                                                  warmup_steps=warmup_steps,
                                                  gamma=gamma)
    train_losses, valid_losses = [], []
else:
    print(f'\nLoading model: {model_name}')
    lr_params = [scheduler_type, learning_rate, lr_decay_time, lr_decay_rate, betas,
                 first_cycle_steps, cycle_mult, max_lr, min_lr, warmup_steps, gamma, betas]
    model, optimizer, scheduler, train_losses, valid_losses = PredNet.load_model(model_name,
                                                                                 n_epochs_run,
                                                                                 epoch_to_load,
                                                                                 lr_params)
    last_epoch = scheduler.last_epoch

# Train the network
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Training network ({n_params} trainable parameters)')
for epoch in range(last_epoch, last_epoch + n_epochs_run):
    print(f'\nRunning epoch {epoch}:')
    train_losses.extend(train_fn(train_dl, model, optimizer, loss_w, remove_ground,
                                 t_start, n_backprop_frames, epoch))
    valid_losses.extend(valid_fn(valid_dl, model, loss_w, remove_ground, t_start, epoch))
    if 'sqm' in dataset_type:
        sqm_losses.append(valid_fn(sqm_dl, model, loss_w, t_start, epoch))
    scheduler.step()
    if (epoch + 1) % n_epochs_save == 0:
        model.save_model(optimizer, scheduler, train_losses, valid_losses)
