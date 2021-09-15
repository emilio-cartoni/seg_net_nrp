import torch
import torch.nn as nn
import numpy as np
import imageio
from src.dataset_fn import VGG_MEAN, VGG_STD
VGG_MEAN = np.array(VGG_MEAN)[None, :, None, None, None]
VGG_STD = np.array(VGG_STD)[None, :, None, None, None]


def train_fn(train_dl, model, optimizer, loss_w, t_start, epoch, plot_gif=True):

  model.train()
  plot_loss_train = 0.0
  n_batches = len(train_dl)
  TA = model.do_time_aligned
  with torch.autograd.set_detect_anomaly(True):
    for batch_idx, (batch, sg_lbl) in enumerate(train_dl):
      batch_loss_train = 0.0
      P_seq, S_seq = [], []
      n_frames = batch.shape[-1]
      for t in range(TA, TA + n_frames):
        A, S_lbl = batch[..., t - TA], sg_lbl[..., t - TA]
        E, P, S = model(A, S_lbl, t - TA)
        P_seq.append(P)
        S_seq.append(S)
        if t >= t_start:
          loss = loss_fn(A, S_lbl, E, P, S, loss_w, batch_idx, n_batches)
          # P_size = P.size()
          # loss = loss_fn(torch.randn(size=P_size).cuda() * P, S_lbl, E, P, S, loss_w, batch_idx, n_batches)
          optimizer.zero_grad()
          loss.backward()
          nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # slowdown?
          optimizer.step()
          model.R_state = [s.detach() for s in model.R_state]
          model.E_state = [s.detach() for s in model.E_state]
          batch_loss_train += loss.detach().item() / n_frames
      plot_loss_train += batch_loss_train / n_batches
      if batch_idx == 0 and plot_gif:
        P_seq = torch.stack(P_seq, axis=-1)
        S_seq = torch.stack(S_seq, axis=-1)
        plot_recons(batch, sg_lbl, P_seq, S_seq, epoch=epoch,
          output_dir=f'./ckpt/{model.model_name}/')
  
  print(f'\r\nEpoch train loss : {plot_loss_train}')
  return plot_loss_train


def valid_fn(valid_dl, model, loss_w, t_start, epoch, plot_gif=True):

  model.eval()
  plot_loss_valid = 0.0
  n_batches = len(valid_dl)
  TA = model.do_time_aligned
  with torch.no_grad():
    for batch_idx, (batch, sg_lbl) in enumerate(valid_dl):
      batch_loss_valid = 0.0
      P_seq, S_seq = [], []
      n_frames = batch.shape[-1]
      for t in range(TA, TA + n_frames):
        A, S_lbl = batch[..., t - TA], sg_lbl[..., t - TA]
        E, P, S = model(A, S_lbl, t - TA)
        P_seq.append(P)
        S_seq.append(S)
        if t >= t_start:
          loss = loss_fn(A, S_lbl, E, P, S, loss_w, batch_idx, n_batches)
          batch_loss_valid += loss.detach().item() / n_frames
      plot_loss_valid += batch_loss_valid / n_batches
      if batch_idx == 0 and plot_gif:
        P_seq = torch.stack(P_seq, axis=-1)
        S_seq = torch.stack(S_seq, axis=-1)
        plot_recons(
          batch, sg_lbl, P_seq, S_seq, epoch=epoch,
          output_dir=f'./ckpt/{model.model_name}/',
          mode='test' if epoch == -1 else 'valid')

  print(f'\r\nEpoch valid loss : {plot_loss_valid}')
  return plot_loss_valid


#bce_loss_fn = nn.BCELoss()
mse_loss_fn = nn.MSELoss()


def loss_fn(frame, S_lbl, E, P, S, loss_w, batch_idx, n_batches):
  zeros = [torch.zeros_like(E[l]) for l in range(len(E))]
  lat_loss = sum([w * (mse_loss_fn(E[l], zeros[l])) for l, w in enumerate(loss_w['lat'])])
  img_loss = mse_loss_fn(P, frame) * loss_w['img']
  seg_loss = mse_loss_fn(S, S_lbl) * loss_w['seg']
  total_loss = lat_loss + img_loss + seg_loss
  print(f'\rBatch ({batch_idx + 1}/{n_batches}) - loss: {total_loss:.3f} ' +
    f'[latent: {lat_loss:.3f}, image: {img_loss:.3f}, segm: {seg_loss:.3f}]', end='')
  return total_loss


def plot_recons(batch, sg_lbl, P_seq, S_seq,
  epoch=0, sample_indexes=(0,), output_dir='./', mode='train'):

  batch_size, n_channels, n_rows, n_cols, n_frames = batch.shape
  img_plot = batch.detach().cpu().numpy() * VGG_STD + VGG_MEAN
  rec_plot = np.clip(P_seq.detach().cpu().numpy() * VGG_STD + VGG_MEAN, 0, 1)
  seg_lbl = onehot_to_rgb(sg_lbl.detach().cpu().numpy())
  seg_plot = np.clip(onehot_to_rgb(S_seq.detach().cpu().numpy()), 0, 1)
  v_rect = np.ones((batch_size, n_channels, n_rows, 10, n_frames))
  data_rec = np.concatenate(
    (img_plot, v_rect, rec_plot, v_rect, v_rect, seg_lbl, v_rect, seg_plot), axis=3)
  out_batch = data_rec.transpose((0, 2, 3, 1, 4))
  for s_idx in sample_indexes:
    out_seq = out_batch[s_idx]
    gif_frames = [(255. * out_seq[..., t]).astype(np.uint8) for t in range(n_frames)]
    gif_path = f'{output_dir}{mode}_epoch{epoch:02}_id{s_idx:02}'
    imageio.mimsave(f'{gif_path}.gif', gif_frames, duration=0.1)


def onehot_to_rgb(onehot_tensor):
  batch_size, n_classes, w, h, n_frames = onehot_tensor.shape
  rgb_tensor = np.zeros((batch_size, 3, w, h, n_frames))
  hue_space = np.linspace(0.0, 1.0, n_classes + 1)[:-1]
  rgb_space = [hsv_to_rgb(hue) for hue in hue_space]
  for n in range(n_classes):
    class_tensor = onehot_tensor[:, n]
    for c, color in enumerate(rgb_space[n]):
      rgb_tensor[:, c] += color * class_tensor
  return rgb_tensor


def hsv_to_rgb(hue):
  v = 1 - abs((int(hue * 360) / 60) % 2 - 1)
  hsv_space = [
    [1, v, 0], [v, 1, 0], [0, 1, v],
    [0, v, 1], [v, 0, 1], [1, 0, v]]
  return hsv_space[int(hue * 6)]
