import torch
import torch.nn as nn
import numpy as np
import imageio
from src.dataset_fn import VGG_MEAN, VGG_STD
VGG_MEAN = VGG_MEAN[None, :, None, None, None]
VGG_STD = VGG_STD[None, :, None, None, None]


def train_fn(train_dl, model, optimizer, loss_w, t_start, epoch, plot_gif=True):

  model.train()
  plot_loss_train = 0.0
  n_batches = len(train_dl)
  with torch.autograd.set_detect_anomaly(True):
    for batch_idx, (batch, sg_lbl) in enumerate(train_dl):
      E_seq, P_seq, S_seq = model(batch)
      loss = loss_fn(
        batch, sg_lbl, E_seq, P_seq, S_seq,
        loss_w, t_start, batch_idx, n_batches)
      optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # slowdown?
      optimizer.step()
      plot_loss_train += loss.detach().item() / n_batches
      if batch_idx == 0 and plot_gif:
        plot_recons(batch, sg_lbl, P_seq, S_seq, epoch=epoch,
          output_dir=f'./ckpt/{model.model_name}/')
  
  print(f'\r\nEpoch train loss : {plot_loss_train}')
  return plot_loss_train


def valid_fn(valid_dl, model, loss_w, t_start, epoch, plot_gif=True):

  model.eval()
  plot_loss_valid = 0.0
  n_batches = len(valid_dl)
  with torch.no_grad():
    for batch_idx, (batch, sg_lbl) in enumerate(valid_dl):
      E_seq, P_seq, S_seq = model(batch)
      loss = loss_fn(
        batch, sg_lbl, E_seq, P_seq, S_seq,
        loss_w, t_start, batch_idx, n_batches)
      plot_loss_valid += loss.detach().item() / n_batches
      if batch_idx == 0 and plot_gif:
        plot_recons(batch, sg_lbl, P_seq, S_seq, epoch=epoch,
          output_dir=f'./ckpt/{model.model_name}/', mode='valid')

  print(f'\r\nEpoch valid loss : {plot_loss_valid}')
  return plot_loss_valid


bce_loss_fn = nn.BCELoss()
mse_loss_fn = nn.MSELoss()
def loss_fn(
  batch, sg_lbl, E_seq, P_seq, S_seq, loss_w, t_start, batch_idx, n_batches):

  times = range(t_start, len(E_seq[0]))
  # lat_loss = sum([w * sum([torch.abs(E_seq[l][t]).mean() for t in times])
  #   for l, w in enumerate(loss_w['lat'])]) / len(times)
  lat_loss = sum([w * sum([E_seq[l][t].mean() for t in times])
    for l, w in enumerate(loss_w['lat'])]) / len(times)

  img_loss = sum([mse_loss_fn(P_seq[..., t], batch[..., t]) for t in times])
  img_loss *= (loss_w['img'] / len(times))

  # seg_loss = sum([bce_loss_fn(S_seq[..., t], sg_lbl[..., t]) for t in times]) * loss_w['seg']
  seg_loss = 0.0
  for t in times:
    inter = (S_seq[..., t] * sg_lbl[..., t]).sum(axis=(-3, -2, -1))  # sum over n_classes, w and h
    union = (S_seq[..., t] + sg_lbl[..., t]).sum(axis=(-3, -2, -1))  # sum over n_classes, w and h
    seg_loss += (1.0 - (2 * inter + 1.0) / (union + 1.0)).mean() * loss_w['seg'] / len(times)

  total_loss = lat_loss + img_loss + seg_loss
  print(f'\rBatch ({batch_idx + 1}/{n_batches}) - loss: {total_loss:.3f} ' +
    f'[latent: {lat_loss:.3f}, image: {img_loss:.3f}, segm: {seg_loss:.3f}]', end='')
  return total_loss


def plot_recons(
  batch, sg_lbl, P_seq, S_seq, epoch=0, sample_indexes=(0,), output_dir='./', mode='train'):

  img_plot = batch.detach().cpu().numpy() * VGG_STD + VGG_MEAN
  rec_plot = P_seq.detach().cpu().numpy() * VGG_STD + VGG_MEAN
  seg_lbl = sg_lbl.detach().cpu().numpy()
  seg_plot = S_seq.detach().cpu().numpy()
  batch_size, n_channels, n_rows, n_cols, n_frames = img_plot.shape
  v_rect = np.ones((batch_size, n_channels, n_rows, 10, n_frames))
  data_rec = np.concatenate(
    (img_plot, v_rect, rec_plot, v_rect, v_rect, seg_lbl, v_rect, seg_plot), axis=3)
  out_batch = data_rec.transpose((0, 2, 3, 1, 4))
  for s_idx in sample_indexes:
    out_seq = out_batch[s_idx]
    gif_frames = [(255. * out_seq[..., t]).astype(np.uint8) for t in range(n_frames)]
    gif_path = f'{output_dir}epoch{epoch:02}_{mode}{s_idx:02}'
    imageio.mimsave(f'{gif_path}.gif', gif_frames, duration=0.1)
