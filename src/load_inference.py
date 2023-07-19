import torch
from src.model import PredNet

def compute_fair_number_of_channels(pred_loss, rnn_type, n_layers):
    ''' Define the layers of the PredNet so that any combination of
        meta-parameters has approximately the same number of parameters
    
    Returns:
    --------
    channels: tuple of int
        Tuple of channels for each layer of PredNet

    '''
    if pred_loss:
        scale = {'hgru': 64, 'conv': 74, 'lstm': 28}[rnn_type]
        channels = tuple([scale * 2 ** i for i in range(n_layers)])
    else:
        scale = {'hgru': 68, 'conv': 96, 'lstm': 30}[rnn_type]
        channels = tuple([scale * 2 ** i for i in range(n_layers)])
    return channels

def load_inference_model(
        ckpt_path, 
        device='cuda' if torch.cuda.is_available() else 'cpu', 
        num_classes=2, 
        rnn_type='hgru',
        axon_delay=True,
        pred_loss=True,
        n_layers=4):

    channels = compute_fair_number_of_channels(pred_loss, rnn_type, n_layers)
    if pred_loss:
        seg_layers = tuple(range(len(channels))[1:])
    else:
        seg_layers = (0,)

    model = PredNet(
        device=device, 
        num_classes=num_classes, 
        rnn_type=rnn_type,
        axon_delay=axon_delay,
        pred_loss=pred_loss,
        seg_layers=seg_layers,
        bu_channels=(3,) + channels[:-1],
        td_channels=channels)
    
    checkpoint = torch.load(ckpt_path)
    states = checkpoint["state_dict"].copy()
    states.clear()
    for k, v in checkpoint["state_dict"].items():
        if not k.startswith("model."):
            print("O")
            continue
        k = k[6:]
        states[k] = v
    model.load_state_dict(states)
    model = model.to(device=device)

    return model, device
