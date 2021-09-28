import torch
import torch.nn as nn
import numpy as np
import imageio

from src.dataset_fn import DATASET_MEAN, DATASET_STD
DATASET_MEAN = np.array(DATASET_MEAN)[None, :, None, None, None]
DATASET_STD = np.array(DATASET_STD)[None, :, None, None, None]

from src.loss_fn import FocalLoss, DiceLoss
bce_loss_fn = nn.BCEWithLogitsLoss()
mse_loss_fn = nn.MSELoss()
mae_loss_fn = nn.L1Loss()
foc_loss_fn = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
dice_loss_fn = DiceLoss()

def train_fn(
  train_dl, model, optimizer, loss_weight, t_start, n_backprop_frames, epoch, plot_gif=True):

  model.train()
  plot_loss_train = 0.0
  n_batches = len(train_dl)
  TA = model.do_time_aligned
  with torch.autograd.set_detect_anomaly(True):
    for batch_idx, (batch, sg_lbl) in enumerate(train_dl):
      batch_loss_train = 0.0
      P_seq, S_seq = [], []
      n_frames = batch.shape[-1]
      loss = 0.0
      for t in range(TA, TA + n_frames):
        A = batch[..., t - TA].to(device='cuda')
        S_lbl = sg_lbl[..., t - TA].to(device='cuda')
        E, P, S = model(A, t - TA)
        P_seq.append(P)
        S_seq.append(S)
        time_weight = float(t >= t_start)
        loss = loss + loss_fn(
          A, S_lbl, E, P, S, time_weight, loss_weight, batch_idx, n_batches)
        if (t - TA + 1) % n_backprop_frames == 0:
          # equivalent to zero_grad(), but faster
          for p in model.parameters(): p.grad = None
          loss.backward()
          nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # slowdown?
          optimizer.step()
          model.A_state = [s.detach() for s in model.A_state]
          model.E_state = [s.detach() for s in model.E_state]
          model.R_state = [s.detach() for s in model.R_state]
          batch_loss_train += loss.detach().item() / n_frames
          loss = 0.0
      plot_loss_train += batch_loss_train / n_batches
      if batch_idx == 0 and plot_gif:
        P_seq = torch.stack(P_seq, axis=-1)
        S_seq = torch.stack(S_seq, axis=-1)
        plot_recons(batch, sg_lbl, P_seq, S_seq, epoch=epoch,
          output_dir=f'./ckpt/{model.model_name}/')
  
  print(f'\r\nEpoch train loss : {plot_loss_train}')
  return plot_loss_train


def valid_fn(valid_dl, model, loss_weight, t_start, epoch, plot_gif=True):

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
        A = batch[..., t - TA].to(device='cuda')
        S_lbl = sg_lbl[..., t - TA].to(device='cuda')
        E, P, S = model(A, t - TA)
        P_seq.append(P)
        S_seq.append(S)
        time_weight = float(t >= t_start)
        loss = loss_fn(
          A, S_lbl, E, P, S, time_weight, loss_weight, batch_idx, n_batches)
        batch_loss_valid += loss.item() / n_frames
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


def loss_fn(frame, S_lbl, E, P, S, time_weight, loss_weight, batch_idx, n_batches):

  # Latent variables prediction loss (unsupervised)
  zeros = [torch.zeros_like(E[l]) for l in range(len(E))]
  latent_loss = sum([w * (mae_loss_fn(E[l], zeros[l])) for l, w in enumerate(loss_weight['latent'])])
  
  # Image prediction loss (unsupervised)
  img_bce_loss = bce_loss_fn(P, frame) * loss_weight['img_bce'] if loss_weight['img_bce'] else 0.0
  img_mae_loss = mae_loss_fn(P, frame) * loss_weight['img_mae'] if loss_weight['img_mae'] else 0.0
  img_mse_loss = mse_loss_fn(P, frame) * loss_weight['img_mse'] if loss_weight['img_mse'] else 0.0
  
  # Segmentation prediction loss (supervised)
  seg_bce_loss = bce_loss_fn(S, S_lbl) * loss_weight['seg_bce'] if loss_weight['seg_bce'] else 0.0
  seg_foc_loss = mae_loss_fn(S, S_lbl) * loss_weight['seg_foc'] if loss_weight['seg_foc'] else 0.0
  seg_mse_loss = mse_loss_fn(S, S_lbl) * loss_weight['seg_mse'] if loss_weight['seg_mse'] else 0.0
  seg_dice_loss = dice_loss_fn(S, S_lbl) * loss_weight['seg_dice'] if loss_weight['seg_dice'] else 0.0
  
  # Total loss
  img_loss = img_bce_loss + img_mae_loss + img_mse_loss
  seg_loss = seg_bce_loss + seg_foc_loss + seg_mse_loss + seg_dice_loss
  total_loss = latent_loss
  if img_loss > 0:
    total_loss = total_loss + img_loss
  if seg_loss > 0:
    total_loss = total_loss + seg_loss
  print(
    f'\rBatch ({batch_idx + 1}/{n_batches}) - loss: {total_loss:.3f} [' +
    f'latent: {latent_loss:.3f}, ' +
    f'image: {img_loss:.3f} (bce: {img_bce_loss:.3f}, ' +
    f'mae: {img_mae_loss:.3f}, mse: {img_mse_loss:.3f}), ' +
    f'segm: {seg_loss:.3f} (bce: {seg_bce_loss:.3f}, ' +
    f'foc: {seg_foc_loss:.3f}, mse: {seg_mse_loss:.3f}, dice: {seg_dice_loss:.3f})' +
    f']', end='')
  return total_loss * time_weight


def plot_recons(batch, sg_lbl, P_seq, S_seq,
  epoch=0, sample_indexes=(0,), output_dir='./', mode='train'):

  batch_size, n_channels, n_rows, n_cols, n_frames = batch.shape
  img_plot = batch.detach().cpu().numpy() * DATASET_STD + DATASET_MEAN
  prediction_plot = P_seq.detach().cpu().numpy() * DATASET_STD + DATASET_MEAN
  seg_lbl = onehot_to_rgb(sg_lbl.detach().cpu().numpy())
  seg_plot = onehot_to_rgb(S_seq.detach().cpu().numpy())
  v_rect = np.ones((batch_size, n_channels, n_rows, 10, n_frames))
  data_rec = np.concatenate(
    (img_plot, v_rect, prediction_plot, v_rect, v_rect, seg_lbl, v_rect, seg_plot), axis=3)
  out_batch = data_rec.transpose((0, 2, 3, 1, 4))
  for s_idx in sample_indexes:
    out_seq = out_batch[s_idx]
    gif_frames = [(255. * out_seq[..., t]).astype(np.uint8) for t in range(n_frames)]
    gif_path = f'{output_dir}{mode}_epoch{epoch:02}_id{s_idx:02}'
    imageio.mimsave(f'{gif_path}.gif', gif_frames, duration=0.1)


def onehot_to_rgb(onehot_tensor):
  batch_size, num_classes, w, h, n_frames = onehot_tensor.shape
  rgb_tensor = np.zeros((batch_size, 3, w, h, n_frames))
  hue_space = np.linspace(0.0, 1.0, num_classes + 1)[:-1]
  rgb_space = [hsv_to_rgb(hue) for hue in hue_space]
  for n in range(num_classes):
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
