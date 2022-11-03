import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from src.hgru_cell import hConvGRUCell
from src.utils import select_scheduler

class PredNet(nn.Module):
    def __init__(
                 self, model_name, do_time_aligned, n_classes, n_layers, td_layers,
                 img_layers, seg_layers, bu_channels, td_channels, device) -> None:
        ''' Create a PredNet model, initialize its states, create a checkpoint
            folder and put the model on the correct device (cpu or gpu).
        
        Parameters
        ----------
        model_name : str
            Name of the model (to identify the checkpoint folder when loading).
        n_classes : int
            Number of classes in the segmentation masks to predict.
        n_layers : int
            Number of layers in the bottom-up and top-down networks.
        td_layers : list of str
            Type of cell used in the top-down computations.
        seg_layers : list of str
            What td_layers are used by the segmentation decoder.
        bu_channels : list of int
            Number of channels in the bottom-up layers.
        td_channels : list of int
            Number of channels in the top-down layers.
        do_segmentation : bool
            Whether to decode segmentation masks.
        device : torch.device
            Device to use for the computation ('cpu', 'cuda').

        Returns
        -------
        None.
        '''
        super(PredNet, self).__init__()

        # Model parameters
        self.model_name = model_name
        self.do_time_aligned = True
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.td_layers = td_layers
        self.img_layers = img_layers
        self.seg_layers = seg_layers
        if bu_channels[0] != 3:
            bu_channels = (3,) + bu_channels[:-1]  # worst coding ever
        self.bu_channels = bu_channels
        self.td_channels = td_channels
        self.do_prediction = not (len(img_layers) == 0)
        self.do_segmentation = not (len(seg_layers) == 0)
        self.device = device
        
        # Model states
        self.E_state = [None for _ in range(n_layers)]
        self.R_state = [None for _ in range(n_layers)]

        # Bottom-up connections (bu)
        bu_conv = []
        for l in range(self.n_layers - 1):  # "2", because error is torch.cat([pos, neg])
            bu_conv.append(nn.Conv2d(2 * bu_channels[l], bu_channels[l + 1], kernel_size=5, padding=2))
        self.bu_conv = nn.ModuleList(bu_conv)
        
        # Lateral connections (la)
        la_conv = []
        for l in range(self.n_layers):
            la_conv.append(nn.Conv2d(td_channels[l], bu_channels[l], kernel_size=1, padding=0))
        self.la_conv = nn.ModuleList(la_conv)

        # Top-down connections (td)
        td_conv = []
        for l in range(self.n_layers):  # "2", because error is torch.cat([pos, neg])
            in_channels = 2 * bu_channels[l] + (td_channels[l + 1] if l < n_layers - 1 else 0)
            if td_layers[l] == 'H':
                td_conv.append(hConvGRUCell(in_channels, td_channels[l], kernel_size=5))  # implicit padding
            elif td_layers[l] == 'C':
                td_conv.append(nn.Conv2d(in_channels, td_channels[l], kernel_size=5, padding=2))
        self.td_conv = nn.ModuleList(td_conv)
        self.td_upsample = nn.ModuleList([nn.Upsample(scale_factor=2) for _ in range(n_layers - 1)])

        # Image segmentation
        if self.do_segmentation:
            input_channels = td_channels  # [2 * b for b in bu_channels]
            self.seg_decoder = Decoder_2D(seg_layers, input_channels, n_classes, nn.Sigmoid())

        # Put model on gpu and create folder for the model
        self.to(device)
        os.makedirs(f'./ckpt/{model_name}/', exist_ok=True)

    def forward(self, A, frame_idx):
        ''' Forward pass of the PredNet.
        
        Parameters
        ----------
        A : torch.Tensor
            Input image (from a batch of input sequences).
        frame_idx : int
            Index of the current frame in the sequence.

        Returns
        -------
        E_pile : list of torch.Tensor
            Activity of all errors units (bottom-up pass).
        img_prediction : torch.Tensor
            Prediction of the next frame input (first layer of the network).
        seg_prediction : torch.Tensor
            Prediction of the segmentation mask (if do_segmentation is True).
        '''
        # Initialize outputs of this step, as well as internal states, if necessary
        batch_dims = A.size()
        batch_size, _, h, w = batch_dims
        E_pile, R_pile = [None] * self.n_layers, [None] * self.n_layers
        if frame_idx == 0:
            for l in range(self.n_layers):
                self.E_state[l] = torch.zeros(batch_size, 2 * self.bu_channels[l], h, w).to(self.device)
                self.R_state[l] = torch.zeros(batch_size, self.td_channels[l], h, w).to(self.device)
                h, w = h // 2, w // 2

        # Bottom-up pass
        for l in range(self.n_layers):
            R_input = self.R_state[l]  # this one is always from last time-step (current expectations)
            A_hat = F.relu(self.la_conv[l](R_input))  # post-synaptic activity of representation prediction
            error = F.relu(torch.cat((A - A_hat, A_hat - A), dim=1))  # post-synaptic activity of A (error goes up)
            E_pile[l] = error  # stored for next step, used in top-down pass
            E_input = self.E_state[l] if self.do_time_aligned else error
            A = F.max_pool2d(E_input, kernel_size=2, stride=2)  # A update for next bu-layer
            if l < self.n_layers - 1:
                A = self.bu_conv[l](A)  # presynaptic activity of bottom-up input
            if l == 0:
                img_prediction = F.hardtanh(A_hat, min_val=0.0, max_val=1.0)

        # Top-down pass
        for l in reversed(range(self.n_layers)):
            td_input = self.E_state[l] if self.do_time_aligned else E_pile[l]
            if l < self.n_layers - 1:
                R_input = self.R_state[l + 1] if self.do_time_aligned else R_pile[l + 1]
                td_output = self.td_upsample[l](R_input)  # here R is input
                td_input = torch.cat((td_input, td_output), dim=1)
            if self.td_layers[l] == 'H':
                R_pile[l] = self.td_conv[l](td_input, self.R_state[l])  # here R is recurrent state
            elif self.td_layers[l] == 'C':
                R_pile[l] = self.td_conv[l](td_input)

        # Image segmentation
        if self.do_segmentation:
            img_segmentation = self.seg_decoder(R_pile)  # E_pile
        else:
            segm_dims = (batch_size, self.n_classes) + batch_dims[2:]
            img_segmentation = torch.zeros(segm_dims).to(self.device)
            
        # Update the states of the network
        self.E_state = E_pile
        self.R_state = R_pile

        # Return the states to the computer
        return E_pile, img_prediction, img_segmentation

    def save_model(self, optimizer, scheduler, train_losses, valid_losses, epoch, n_epochs_save):
        ''' Save the model and the training history.
        
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer used to train the model.
        scheduler : torch.optim.lr_scheduler
            Learning rate scheduler used to train the model.
        train_losses : list of float
            Training losses of the model.
        valid_losses : list of float
            Validation losses of the model.

        Returns
        -------
        None
        '''
        ckpt_id = epoch // n_epochs_save * n_epochs_save
        torch.save({
            'model_name': self.model_name,
            'do_time_aligned': self.do_time_aligned,
            'n_classes': self.n_classes,
            'n_layers': self.n_layers,
            'td_layers': self.td_layers,
            'img_layers': self.img_layers,
            'seg_layers': self.seg_layers,
            'bu_channels': self.bu_channels,
            'td_channels': self.td_channels,
            'device': self.device,
            'model_params': self.state_dict(),
            'optimizer_params': optimizer.state_dict(),
            'scheduler_params': scheduler.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses},
            rf'.\ckpt\{self.model_name}\ckpt_{ckpt_id:03}.pt')
        print('SAVED')
        train_losses = [l if l < 10 * sum(train_losses) / len(train_losses) else 0.0 for l in train_losses]
        valid_losses = [l if l < 10 * sum(valid_losses) / len(valid_losses) else 0.0 for l in valid_losses]
        train_axis = list(np.arange(0, epoch + 1, (epoch + 1) / len(train_losses)))[:len(train_losses)]
        valid_axis = list(np.arange(0, epoch + 1, (epoch + 1) / len(valid_losses)))[:len(valid_losses)]
        plt.plot(train_axis, train_losses, label='train')
        plt.plot(valid_axis, valid_losses, label='valid')
        plt.legend()
        plt.savefig(rf'.\ckpt\{self.model_name}\loss_plot.png')
        plt.close()

    @classmethod
    def load_model(cls, model_name, epoch_to_load=None, lr_params=None):
        ''' Load a model from a checkpoint.
        
        Parameters
        ----------
        model_name : str
            Name of the model (used for retrieve the checkpoint folder).
        n_epochs_run : int
            Number of epochs the model has been trained on.
        epoch_to_load : int
            Epoch to load a checkpoint from.
        lr_params : dict
            Learning rate parameters (for optimizer and scheduler).

        Returns
        -------
        model : Model
            Loaded model.
        '''
        ckpt_dir = f'./ckpt/{model_name}/'
        list_dir = sorted([c for c in os.listdir(ckpt_dir) if '.pt' in c])
        ckpt_path = list_dir[-1]  # take last checkpoint (default)
        for ckpt in list_dir:
            if str(epoch_to_load) in ckpt.split('_')[-1]:
                ckpt_path = ckpt
        save = torch.load(ckpt_dir + ckpt_path)
        model = cls(
            model_name=model_name,
            n_classes=save['n_classes'],
            do_time_aligned=save['do_time_aligned'],
            n_layers=save['n_layers'],
            td_layers=save['td_layers'],
            img_layers=save['img_layers'],
            seg_layers=save['seg_layers'],
            bu_channels=save['bu_channels'],
            td_channels=save['td_channels'],
            device=save['device'])
        model.load_state_dict(save['model_params'])
        valid_losses = save['valid_losses']
        train_losses = save['train_losses']
        if lr_params is None:
            optimizer, scheduler = None, None
        else:
            optimizer = torch.optim.AdamW(model.parameters(), **lr_params['optimizer'])
            scheduler = select_scheduler(optimizer, lr_params)
            optimizer.load_state_dict(save['optimizer_params'])
            scheduler.load_state_dict(save['scheduler_params'])
        return model, optimizer, scheduler, train_losses, valid_losses


class Decoder_2D_bis(nn.Module):
    def __init__(self, decoder_layers, input_channels, output_channels, output_fn):
        ''' Decoder for any set of 2D images (e.g., segmentation masks).
            This version is in development and does not work yet.
        
        Parameters
        ----------
        decoder_layers : list of int
            Layers of PredNet from which the 2D labels are decoded.
        input_channels : list of int
            Number of channels of in the decoded layers.
        output_channels : int
            Number of channels of the output to decode.
        output_fn : str
            Activation function of the decoder.

        Returns
        -------
        None
        '''
        super(Decoder_2D_bis, self).__init__()
        self.decoder_layers = decoder_layers
        hidden_channels = 64
        hidden_channels_out = hidden_channels * len(decoder_layers)
        up_conv = []
        for l in decoder_layers:
            to_append = [] if l > 0 else [nn.Conv2d(input_channels[l], hidden_channels, 1)]
            for s in range(l):
                in_channels = input_channels[l] if s == 0 else hidden_channels
                to_append.append(nn.ConvTranspose2d(in_channels, hidden_channels, 2, stride=2))
            up_conv.append(nn.Sequential(*to_append))
        self.up_conv = nn.ModuleList(up_conv)
        self.out_conv = nn.Sequential(nn.Conv2d(hidden_channels_out, output_channels, 1), output_fn)
        
    def forward(self, R_pile):
        ''' Forward pass of the decoder.
        
        Parameters
        ----------
        R_pile : list of torch.Tensor
            List of tensors of shape (batch_size, channels, height, width)
            containing the activity of the latent units of the PredNet.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, channels, height, width)
            containing the decoded output.
        '''
        hidden_input = [self.up_conv[i](R_pile[l]) for i, l in enumerate(self.decoder_layers)]
        hidden_input = F.relu(torch.cat(hidden_input, dim=1))
        return self.out_conv(hidden_input)


class Decoder_2D(nn.Module):
    def __init__(self, decoder_layers, input_channels, n_output_channels, output_fn):
        ''' Decoder for any set of 2D images (e.g., segmentation masks).
            This version is in development and does not work yet.
        
        Parameters
        ----------
        decoder_layers : list of int
            Layers of PredNet from which the 2D labels are decoded.
        input_channels : list of int
            Number of channels of in the decoded layers.
        n_output_channels : int
            Number of channels of the output to decode.
        output_fn : str
            Activation function of the decoder.

        Returns
        -------
        None
        '''
        super(Decoder_2D, self).__init__()
        self.decoder_layers = decoder_layers
        decoder_upsp = []
        for l in range(1, max(decoder_layers) + 1):
            inn, out = input_channels[l], input_channels[l - 1]
            decoder_upsp.append(nn.Sequential(
                nn.GroupNorm(inn, inn),
                nn.ConvTranspose2d(in_channels=inn, out_channels=out,
                    kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.GELU()))
        self.decoder_upsp = nn.ModuleList([None] + decoder_upsp)  # None for indexing convenience
        self.decoder_conv = nn.Sequential(
            nn.GroupNorm(input_channels[0], input_channels[0]),
            nn.Conv2d(input_channels[0], n_output_channels, kernel_size=1, bias=False),
            output_fn)
  
    def forward(self, R_pile):
        ''' Forward pass of the decoder.
        
        Parameters
        ----------
        R_pile : list of torch.Tensor
            List of tensors of shape (batch_size, channels, height, width)
            containing the activity of the latent units of the PredNet.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, channels, height, width)
            containing the decoded output.
        '''
        D = R_pile[max(self.decoder_layers)]
        D = self.decoder_upsp[-1](D) if max(self.decoder_layers) > 0 else self.decoder_conv(D)
        for l in reversed(range(max(self.decoder_layers))):
            if l in self.decoder_layers:
                D = D + R_pile[l]
            D = self.decoder_upsp[l](D) if l > 0 else self.decoder_conv(D)
        return D
