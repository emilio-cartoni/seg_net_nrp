import torch
import torch.nn as nn
import imageio
import numpy as np
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from src.loss_fn import FocalLoss, DiceLoss

bce_loss_fn = nn.BCEWithLogitsLoss()
mse_loss_fn = nn.MSELoss()
mae_loss_fn = nn.L1Loss()
foc_loss_fn = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
# dic_loss_fn = DiceLoss(weight=[0.5, 1.0, 5.0, 1.0])
# dic_loss_fn = DiceLoss(weight=[1.0, 5.0, 1.0])
# dic_loss_fn = DiceLoss(weight=[10.0, 40.0, 70.0, 1.0])  # for handover dataset including the torso
# dic_loss_fn = DiceLoss(weight=[1.0, 10.0, 10.0, 50.0, 50.0])  # 'torso', 'upper_arm', 'forearm', 'hand', 'tool'
dic_loss_fn = DiceLoss()

def train_fn(train_dl, model, optimizer, scheduler, loss_params, epoch, plot_gif=True):

    # Train the network for one epoch
    model.train()
    TA = model.do_time_aligned
    plot_loss_train = []  # 0.0
    n_batches = len(train_dl)
    with torch.autograd.set_detect_anomaly(True):

        for batch_idx, (images, S_lbls) in enumerate(train_dl):
            batch_loss_train = 0.0
            A_seq, P_seq, S_seq, S_lbl_seq = [], [], [], []
            n_frames = images.shape[-1]
            loss = 0.0

            for t in range(TA, TA + n_frames):
                A = images[..., t - TA].to(device=model.device)
                S_lbl = S_lbls[..., t - TA].to(device=model.device)
                E, P, S = model(A, t - TA)
                A_seq.append(A.detach().cpu())
                P_seq.append(P.detach().cpu())
                S_seq.append(S.detach().cpu())
                S_lbl_seq.append(S_lbl.detach().cpu())
                time_weight = float(t - TA >= loss_params['t_start'])  # 0.0 if t < loss_params['t_start'] else (1.0 / t)
                loss = loss + loss_fn(E, A, P, S, S_lbl, time_weight, loss_params, batch_idx, n_batches)

                if (t - TA + 1) % loss_params['n_backprop_frames'] == 0:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # slowdown?
                    optimizer.step()
                    if model.do_time_aligned:
                        model.E_state = [s.detach() if s is not None else s for s in model.E_state]
                        model.R_state = [s.detach() if s is not None else s for s in model.R_state]
                    batch_loss_train += loss.detach().item() / n_frames
                    loss = 0.0
                
            plot_loss_train.append(batch_loss_train)  # += batch_loss_train / n_batches
            if ((epoch == 0 and (batch_idx % 10) == 0) or (batch_idx == 0)) and plot_gif:
                A_seq = torch.stack(A_seq, axis=-1)
                P_seq = torch.stack(P_seq, axis=-1)
                S_seq = torch.stack(S_seq, axis=-1)
                S_lbl_seq = torch.stack(S_lbl_seq, axis=-1)
                plot_recons(A_seq, S_lbl_seq, P_seq, S_seq,
                    epoch=epoch, batch_idx=batch_idx,
                    output_dir=f'./ckpt/{model.model_name}/')
            
            if type(scheduler) is torch.optim.lr_scheduler.OneCycleLR:
                scheduler.step()
            elif batch_idx == (len(train_dl) - 1):
                scheduler.step()
                
    print(f'\r\nEpoch train loss : {sum(plot_loss_train) / len(plot_loss_train)}')
    return plot_loss_train


def valid_fn(valid_dl, model, loss_params, epoch, plot_gif=True):

    model.eval()
    TA = model.do_time_aligned
    plot_loss_valid = []  # 0.0
    n_batches = len(valid_dl)
    with torch.no_grad():

        for batch_idx, (images, S_lbls) in enumerate(valid_dl):
            batch_loss_valid = 0.0
            A_seq, P_seq, S_seq, S_lbl_seq = [], [], [], []
            n_frames = images.shape[-1]

            for t in range(TA, TA + n_frames):
                A = images[..., t - TA].to(device=model.device)
                S_lbl = S_lbls[..., t - TA].to(device=model.device)
                E, P, S = model(A, t - TA)
                A_seq.append(A.detach().cpu())
                P_seq.append(P.detach().cpu())
                S_seq.append(S.detach().cpu())
                S_lbl_seq.append(S_lbl.detach().cpu())
                time_weight = float(t - TA >= loss_params['t_start'])  # 0.0 if t < loss_params['t_start'] else (1.0 / t)
                loss = loss_fn(E, A, P, S, S_lbl, time_weight, loss_params, batch_idx, n_batches)
                batch_loss_valid += loss.item() / n_frames
                
            plot_loss_valid.append(batch_loss_valid)  # += batch_loss_valid / n_batches
            if ((epoch == 0 and (batch_idx % 10) == 0) or (batch_idx == 0)) and plot_gif:
                A_seq = torch.stack(A_seq, axis=-1)
                P_seq = torch.stack(P_seq, axis=-1)
                S_seq = torch.stack(S_seq, axis=-1)
                S_lbl_seq = torch.stack(S_lbl_seq, axis=-1)
                plot_recons(
                    A_seq, S_lbl_seq, P_seq, S_seq,
                    epoch=epoch, batch_idx=batch_idx,
                    output_dir=f'./ckpt/{model.model_name}/',
                    mode='test' if epoch == -1 else 'valid')

    print(f'\r\nEpoch valid loss: {sum(plot_loss_valid) / len(plot_loss_valid)}\n')
    return plot_loss_valid


def loss_fn(E, frame, P, S, S_lbl, time_weight, loss_params, batch_idx, n_batches):

    # Latent prediction error loss (unsupervised)
    latent = 0.0 if E is None else sum([torch.mean(E[l]) * w for l, w in enumerate(loss_params['latent'])])
    
    # Image prediction loss (unsupervised)
    img_mae_loss = mae_loss_fn(P, frame) * loss_params['prd_mae'] if loss_params['prd_mae'] else 0.0
    img_mse_loss = mse_loss_fn(P, frame) * loss_params['prd_mse'] if loss_params['prd_mse'] else 0.0

    # Segmentation prediction loss (supervised)
    if loss_params['remove_ground']:  # ensure segmentation has background class
        S = torch.cat([1.0 - torch.sigmoid(S.sum(axis=1, keepdim=True)), S], axis=1)
        S_lbl = torch.cat([1.0 - S_lbl.sum(axis=1, keepdim=True), S_lbl], axis=1)
    seg_foc_loss = foc_loss_fn(S, S_lbl) * loss_params['seg_foc'] if loss_params['seg_foc'] else 0.0
    seg_dic_loss = dic_loss_fn(S, S_lbl) * loss_params['seg_dic'] if loss_params['seg_dic'] else 0.0
   
    # Total loss
    img_loss = img_mae_loss + img_mse_loss
    seg_loss = seg_foc_loss + seg_dic_loss
    total_loss = latent + img_loss + (seg_loss if seg_loss > 0 else 0.0)
    print(
        f'\rBatch ({batch_idx + 1}/{n_batches}) - loss: {total_loss:.3f} [latent: {latent:.3f}, ' +
        f'image: {img_loss:.3f} (mae: {img_mae_loss:.3f}, mse: {img_mse_loss:.3f}) ' +
        f'seg: {seg_loss:.3f} (foc: {seg_foc_loss:.3f}, dic: {seg_dic_loss:.3f})]', end='')
    return total_loss * time_weight


def plot_recons(A_seq, S_lbl_seq, P_seq, S_seq, epoch=0, batch_idx=(0,), output_dir='./', mode='train'):

    batch_size, n_channels, n_rows, n_cols, n_frames = A_seq.shape
    img_plot = A_seq.numpy()
    pred_plot = P_seq.numpy()
    seg_lbl_plot = onehot_to_rgb(S_lbl_seq.numpy())
    seg_plot = onehot_to_rgb(S_seq.numpy())
    
    rect_width = 10
    h_rect = np.ones((batch_size, n_channels, rect_width, n_cols, n_frames))
    v_rect = np.ones((batch_size, n_channels, 2 * n_rows + rect_width, rect_width, n_frames))
    img_data = np.concatenate((img_plot, h_rect, pred_plot), axis=2)
    loc_data = np.concatenate((seg_lbl_plot, h_rect, seg_plot), axis=2)
    out_batch = np.concatenate((img_data, v_rect, loc_data), axis=3)
    out_batch = out_batch.transpose((0, 2, 3, 1, 4))

    out_seq = out_batch[0]
    gif_frames = [(255. * out_seq[..., t]).astype(np.uint8) for t in range(n_frames)]
    gif_path = f'{output_dir}{mode}_epoch_{epoch:03}_id_{batch_idx:03}'
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


def select_scheduler(optimizer, lr_params):
    if lr_params['scheduler_type'] == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **lr_params['multistep'])
    elif lr_params['scheduler_type'] == 'cosine':
        return CosineAnnealingWarmupRestarts(optimizer, **lr_params['cosine'])
    elif lr_params['scheduler_type'] == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **lr_params['onecycle'])
        if scheduler.last_epoch != -1:  # not sure if necessary
            lr_params['onecycle']['last_epoch'] = scheduler.last_epoch
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **lr_params['onecycle'])
        return scheduler
