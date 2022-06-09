import torch
import numpy as np
import imageio
from src.model import PredNet
from src.dataset_fn_handover import get_handover_dataloaders
from src.dataset_fn_multi import get_multi_dataloaders

def main():
    ''' Test the PredNet model on a sequence of images.'''
    
    # Load model
    device = 'cuda'
    remove_ground = True
    model_name = 'Michael_ckpt'
    model, _, _, _, _ = PredNet.load_model(model_name)
    model.eval()
    model.to(device)

    # Dataset parameters
    data_params = {
        'batch_size_train': 4,
        'batch_size_valid': 16,
        'n_frames': 50,
        'tr_ratio': 0.8,
        'remove_ground': remove_ground,
        'augmentation': True,
        'dataset_dir': 'handover',  # 'mots', 'nrp', 'handover', 'multi_shelf', 'multi_small', 'bmw'
        'dataset_path': {
            'handover': r'D:\DL\datasets\nrp\handover',
            'multi_shelf': r'D:\DL\datasets\nrp\multi_shelf',
            'multi_small': r'D:\DL\datasets\nrp\multi_small'}}
    dataloader_fn = {  # TODO: either code them or make only one file loader
        'handover': get_handover_dataloaders,
        'multi_shelf': get_multi_dataloaders,
        'multi_small': get_multi_dataloaders}[data_params['dataset_dir']]
    train_dl, valid_dl, n_classes = dataloader_fn(**data_params)
    
    # Main run loop
    n_samples_to_plot = 1
    seg_pred_sequence = []
    with torch.no_grad():
        for batch_idx, (input_sequence, label_sequence) in enumerate(valid_dl):

            # Run the segmentation model model
            if batch_idx < n_samples_to_plot:
                for t in range(input_sequence.shape[-1]):
                    image = input_sequence[..., t].to(device=device)
                    _, _, seg_pred = model(image, t)
                    seg_pred_numpy = seg_pred.cpu().numpy()
                    seg_pred_sequence.append(seg_pred_numpy)
    
                # Plot the segmentation prediction
                plot_seg_pred_sequence(seg_pred_sequence, batch_idx)
            
            # Stop after n_samples_to_plot samples
            else:
                break


def plot_seg_pred_sequence(seg_pred_sequence, batch_idx):
    ''' Plots a sequence of segmentation images.
    
    Parameters
    ----------
    seg_image_sequence : np.array
        Array of shape (batch_size, 3, height, width, n_frames)
        containing the segmentation images.
    batch_idx : int
        Index of the batch being plotted.
    
    Returns
    -------
    None
    '''
    n_frames = len(seg_pred_sequence)
    seg_pred_stack = np.stack(seg_pred_sequence, axis=-1)
    seg_pred_rgb = onehot_to_rgb(seg_pred_stack).transpose(0, 2, 3, 1, 4)
    seg_pred_rgb = (255 * seg_pred_rgb).astype(np.uint8)
    for sample_idx, sample in enumerate(seg_pred_rgb):
        sample_list = [sample[..., t] for t in range(n_frames)]
        gif_path = f'./seg_output_batch{batch_idx:03}_sample_{sample_idx:03}.gif'
        imageio.mimsave(gif_path,
                        sample_list,
                        duration=0.1)


def onehot_to_rgb(onehot_array):
    ''' Converts a one-hot encoded array to an RGB array.
    
    Parameters
    ----------
    onehot_array : np.array
        Array of shape (batch_size, height, width, n_frames, n_classes)
        containing the one-hot encoded images.

    Returns
    -------
    rgb_array : np.array
        Array of shape (batch_size, height, width, n_frames, 3)
        containing the RGB images.
    '''
    batch_size, num_classes, w, h, n_frames = onehot_array.shape
    rgb_array = np.zeros((batch_size, 3, w, h, n_frames))
    hue_space = np.linspace(0.0, 1.0, num_classes + 1)[:-1] 
    rgb_space = [hsv_to_rgb(hue) for hue in hue_space]
    for n in range(num_classes):
        class_array = onehot_array[:, n]
        for c, color in enumerate(rgb_space[n]):
            rgb_array[:, c] += color * class_array
    return rgb_array


def hsv_to_rgb(hue):
    ''' Converts a hue value to an RGB color.
    
    Parameters
    ----------
    hue : float
        Hue value in the range [0.0, 1.0].

    Returns
    -------
    rgb : np.array
        Array of shape (3,) containing the RGB color.
    '''
    v = 1 - abs((int(hue * 360) / 60) % 2 - 1)
    hsv_space = [
        [1, v, 0], [v, 1, 0], [0, 1, v],
        [0, v, 1], [v, 0, 1], [1, 0, v]]
    return hsv_space[int(hue * 6)]


if __name__ == '__main__':
    main()
