import torch
import numpy as np
import torch.nn.functional as F


def plot_recons(P_seq, P_seq_true, S_seq, S_seq_true, rect_width=10):
    ''' Plot reconstruction of image and segmentation mask. '''
    rect_width=320
    N, C, H, W, T = P_seq.shape
    S_seq = onehot_to_rgb(S_seq.numpy())
    S_seq_true = onehot_to_rgb(S_seq_true.numpy())
    
    h_rect = torch.ones((N, C, rect_width, W, T))
    v_rect = torch.ones((N, C, 2 * H + rect_width, rect_width, T))
    img_data = torch.cat((P_seq_true, h_rect, P_seq), dim=2)
    seg_data = torch.cat((S_seq_true, h_rect, S_seq), dim=2)
    
    gif_batch = torch.cat((img_data, v_rect, seg_data), dim=3)
    gif_batch = gif_batch.permute(0, 4, 1, 2, 3)  # (N, T, C, H, W)
    pad_tuple = (rect_width, rect_width, rect_width, rect_width)
    gif_batch = F.pad(gif_batch, pad_tuple, value=0.5)
    return gif_batch


def onehot_to_rgb(onehot_array):
    batch_size, num_classes, w, h, n_frames = onehot_array.shape
    rgb_array = np.zeros((batch_size, 3, w, h, n_frames))
    hue_space = np.linspace(0.0, 1.0, num_classes + 1)[:-1]
    rgb_space = [hsv_to_rgb(hue) for hue in hue_space]
    for n in range(num_classes):
        class_array = onehot_array[:, n]
        for c, color in enumerate(rgb_space[n]):
            rgb_array[:, c] += color * class_array
    rgb_tensor = torch.from_numpy(rgb_array)
    return rgb_tensor


def hsv_to_rgb(hue):
    v = 1 - abs((int(hue * 360) / 60) % 2 - 1)
    hsv_space = [
        [1, v, 0], [v, 1, 0], [0, 1, v],
        [0, v, 1], [v, 0, 1], [1, 0, v]]
    return hsv_space[int(hue * 6)]


def select_scheduler(optimizer, scheduler_type, lr, num_epochs, num_batches):
    lr_params = {
        'multistep': {
            'milestones': range(5, 10000, 5),
            'gamma': 0.75},
        'cosine': {
            'first_cycle_steps': 10,
            'cycle_mult': 1.0,
            'max_lr': lr,
            'min_lr': lr / 100,
            'warmup_steps': 2,
            'gamma': 1.0},
        'onecycle': {
            'max_lr': lr,
            'steps_per_epoch': num_batches + 1,
            'epochs': num_epochs + 1},
        'exp':{'gamma': 1.2}}
    if scheduler_type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            **lr_params['multistep']
        )
    elif scheduler_type == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            **lr_params['onecycle']
        )
    elif scheduler_type == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            **lr_params['exp']
        )
    else:
        raise ValueError('Scheduler type not recognized.')
