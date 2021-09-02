import torch
from src.model import PredNetVGG
from src.utils import train_fn, valid_fn
from src.dataset_fn import get_datasets_seg

# Parameters
load_model, n_epochs_run, n_epoch_save, epoch_to_load = False, 50, 2, None
do_time_aligned, do_untouched_bu, do_bens_idea = True, False, True
batch_size_train, batch_size_valid = 1, 4
vgg_type, n_layers, t_start = 'vgg19', 4, 1
pr_layers = tuple([l for l in [0, 1, 2, 3] if l < n_layers])  # set as [] for not doing it
sg_layers = tuple([l for l in [0, 1, 2, 3] if l < n_layers])  # set as [] for not doing it
td_channels = td_channels = (32, 64, 128, 256) if do_bens_idea else (64, 128, 256, 512)
td_channels = td_channels[:n_layers]
learning_rate, dropout_rates = 1e-5, (0.0, 0.0, 0.0, 0.0)[:n_layers]
loss_w = {
  'lat': (0.1, 0.1, 0.1, 0.1)[:n_layers],
  'img': 100.0 if len(pr_layers) > 0 else 0.0,
  'seg': 100.0 if len(sg_layers) > 0 else 0.0}
model_name = f'{vgg_type}_TA{int(do_time_aligned)}_BU{int(do_untouched_bu)}'\
           + f'_BI{int(do_bens_idea)}_TD{td_channels}_PR{pr_layers}_SG{sg_layers}'\
           + f'_DR{tuple([int(10 * r) for r in dropout_rates])}'
model_name = model_name.replace('.', '-').replace(',', '-').replace(' ', '').replace("'", '')

# Dataset
dataset_path = 'D:/DL/datasets/nrp/training_room_dataset_00.h5'
n_samples, tr_ratio = 1000, 0.85  # n_train(valid)_samples = ((1-)tr_ratio) * n_samples
augmentation, remove_ground, speedup_factor = True, True, 1
n_classes = 3 if remove_ground else 4
train_dl, valid_dl = get_datasets_seg(
  dataset_path, tr_ratio, n_samples, batch_size_train, batch_size_valid,
  augmentation=augmentation, remove_ground=remove_ground, speedup_factor=speedup_factor)

# Load the model
if not load_model:
  print(f'\nCreating model: {model_name}')
  model = PredNetVGG(
    model_name, vgg_type, n_classes, n_layers, pr_layers, sg_layers, \
    td_channels, dropout_rates, do_time_aligned, do_untouched_bu, do_bens_idea)
  train_losses, valid_losses, last_epoch = [], [], 0
  optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=0.1)
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
  train_losses.append(train_fn(train_dl, model, optimizer, loss_w, t_start, epoch))
  valid_losses.append(valid_fn(valid_dl, model, loss_w, t_start, epoch))
  scheduler.step()
  if (epoch + 1) % n_epoch_save == 0:
    model.save_model(optimizer, scheduler, train_losses, valid_losses)
