import torch
import torch.nn as nn
import torch.nn.functional as F
from src.convlstm_cell import ConvLSTMCell
from src.hgru_cell import hConvGRUCell

class PredNet(nn.Module):
    def __init__(self, device, num_classes, rnn_type, axon_delay,
                 pred_loss, seg_layers, bu_channels, td_channels) -> None:
        ''' Create a PredNet model, initialize its states, create a checkpoint
            folder and put the model on the correct device (cpu or gpu).
        
        Args:
        -----
        device : torch.device
            Device on which the model will be trained (cpu or gpu).
        num_classes : int
            Number of classes in the segmentation masks to predict.
        rnn_type : str
            Type of the recurrent cells used in the model.
        axon_delay : bool
            Whether or not to use axonal delays in the model.
        pred_loss: bool
            Whether or not to minimize prediction error in the model.
        seg_layers : list of str
            What td_layers are used by the segmentation decoder.
        bu_channels : list of int
            Number of channels in the bottom-up layers.
        td_channels : list of int
            Number of channels in the top-down layers.

        '''
        super(PredNet, self).__init__()

        # Model parameters
        self.device = device
        self.n_classes = num_classes
        self.rnn_type = rnn_type
        self.axon_delay = axon_delay
        self.pred_loss = pred_loss
        self.n_layers = len(td_channels)
        self.bu_channels = bu_channels
        self.td_channels = td_channels
        self.do_segmentation = not (len(seg_layers) == 0)
        
        # Model states
        self.E_state = [None for _ in range(self.n_layers)]
        self.R_state = [None for _ in range(self.n_layers)]

        # Bottom-up connections (bu)
        bu_conv = []
        for l in range(self.n_layers - 1):
            in_channels = (2 if pred_loss else 1) * bu_channels[l]
            bu_conv.append(nn.Conv2d(in_channels,
                                     bu_channels[l + 1],
                                     kernel_size=5,
                                     padding=2))
        self.bu_conv = nn.ModuleList(bu_conv)
        
        # Lateral connections (la)
        la_conv = []
        for l in range(self.n_layers):
            la_conv.append(nn.Conv2d(td_channels[l],
                                     bu_channels[l],
                                     kernel_size=1,
                                     padding=0))
        self.la_conv = nn.ModuleList(la_conv)

        # Top-down connections (td)
        td_conv, td_upsample = [], []
        for l in range(self.n_layers):
            in_channels = (2 if pred_loss else 1) * bu_channels[l]
            if l < self.n_layers - 1:
                in_channels = in_channels + td_channels[l + 1]
            if rnn_type == 'hgru':
                td_conv.append(hConvGRUCell(in_channels,
                                            td_channels[l],
                                            kernel_size=5))
            elif rnn_type == 'lstm':
                td_conv.append(ConvLSTMCell(in_channels=in_channels,
                                            out_channels=td_channels[l],
                                            kernel_size=5,
                                            padding=2))
            elif rnn_type == 'conv':
                td_conv.append(nn.Conv2d(in_channels,
                                         td_channels[l],
                                         kernel_size=5,
                                         padding=2))
            if l < self.n_layers - 1:
                td_upsample.append(nn.Upsample(scale_factor=2))
                # td_upsample.append(nn.ConvTranspose2d(td_channels[l + 1],
                #                                       td_channels[l + 1],
                #                                       kernel_size=2,
                #                                       stride=2)
        self.td_conv = nn.ModuleList(td_conv)
        self.td_upsample = nn.ModuleList(td_upsample)

        # Image segmentation
        if self.do_segmentation:
            input_channels = td_channels
            self.seg_decoder = Decoder_2D(seg_layers,
                                          input_channels,
                                          self.n_classes,
                                          nn.Sigmoid())
    
    def init_states(self, batch_size, h, w, t):

        # Initialize model states at start or during batch size change
        if self.E_state[0] is None or self.E_state[0].shape[0] != batch_size:
            for l in range(self.n_layers):
                e_channels = (2 if self.pred_loss else 1) * self.bu_channels[l]
                E_state_shape = (batch_size, e_channels, h, w)
                R_state_shape = (batch_size, self.td_channels[l], h, w)
                self.E_state[l] = torch.zeros(*E_state_shape).to(self.device)
                self.R_state[l] = torch.zeros(*R_state_shape).to(self.device)
                if self.rnn_type == 'lstm':
                    self.R_state[l] = torch.zeros(R_state_shape).to(self.device)
                h, w = h // 2, w // 2

        # Cut gradient computation for the states when kept between batches
        elif t == 0:
            self.E_state = [s.detach() if s is not None else s for s in self.E_state]
            self.R_state = [s.detach() if s is not None else s for s in self.R_state]
                
    def forward(self, A, t):
        ''' Forward pass of the PredNet.
        
        Args:
        -----
        A : torch.Tensor
            Input image (from a batch of input sequences).

        Returns:
        --------
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
        if self.rnn_type == 'lstm':
            L_pile = [None] * self.n_layers
        self.init_states(batch_size, h, w, t)

        # Bottom-up pass
        for l in range(self.n_layers):
            R_input = self.R_state[l]
            A_hat = F.relu(self.la_conv[l](R_input))  # post-synaptic activity of representation prediction
            if self.pred_loss:
                signal = F.relu(torch.cat((A - A_hat, A_hat - A), dim=1))  # prediction error signal
            else:
                signal = F.relu(A)  # normal signal (prediction error not computed)
            E_pile[l] = signal  # error is stored for next step, used in top-down pass
            E_input = self.E_state[l] if self.axon_delay else signal  # using last time-step or not
            if l < self.n_layers - 1:
                E_input = self.bu_conv[l](E_input)  # pre-synaptic activity of next layer (not for input image)
            if l == 0:
                img_prediction = F.hardtanh(A_hat, min_val=0.0, max_val=1.0)
            A = F.max_pool2d(E_input, kernel_size=2, stride=2)  # A update for next bu-layer

        # Top-down pass
        for l in reversed(range(self.n_layers)):
            td_input = self.E_state[l] if self.axon_delay else E_pile[l]
            if l < self.n_layers - 1:
                R_input = self.R_state[l + 1] if self.axon_delay else R_pile[l + 1]
                td_output = self.td_upsample[l](R_input)
                td_input = torch.cat((td_input, td_output), dim=1)
            if self.rnn_type == 'hgru':
                R_pile[l] = self.td_conv[l](td_input, self.R_state[l])
            if self.rnn_type == 'lstm':
                R_pile[l], L_pile[l] = self.td_conv[l](td_input,
                                                       self.R_state[l],
                                                       self.R_state[l])
            elif self.rnn_type == 'conv':
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
        if self.rnn_type == 'lstm':
            self.L_state = L_pile

        # Return the states to the computer
        return E_pile, img_prediction, img_segmentation

  
class Decoder_2D(nn.Module):
    def __init__(self, decoder_layers, input_channels, n_output_channels, output_fn):
        ''' Decoder for any set of 2D images (e.g., segmentation masks).
            Combines different pooling levels like in Long et al. (2015).
        
        Args:
        -----
        decoder_layers : list of int
            Layers of PredNet from which the 2D labels are decoded.
        input_channels : list of int
            Number of channels of in the decoded layers.
        n_output_channels : int
            Number of channels of the output to decode.
        output_fn : str
            Activation function of the decoder.

        '''
        super(Decoder_2D, self).__init__()
        self.decoder_layers = decoder_layers
        decoder_upsp = []
        for l in range(1, max(decoder_layers) + 1):
            inn, out = input_channels[l], input_channels[l - 1]
            decoder_upsp.append(nn.Sequential(
                nn.GroupNorm(inn, inn),
                nn.ConvTranspose2d(in_channels=inn,
                                   out_channels=out,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1,
                                   bias=False),
                nn.ReLU()))
        self.decoder_upsp = nn.ModuleList([None] + decoder_upsp)  # None for indexing convenience
        self.decoder_conv = nn.Sequential(
            nn.GroupNorm(input_channels[0], input_channels[0]),
            nn.Conv2d(input_channels[0],
                      n_output_channels,
                      kernel_size=1,
                      bias=False),
            output_fn)
  
    def forward(self, R_pile):
        ''' Forward pass of the decoder.
            Combines different pooling levels like in Long et al. (2015)
        
        Args:
        -----
        R_pile : list of torch.Tensor
            List of tensors of shape (batch_size, channels, height, width)
            containing the activity of the latent units of the PredNet.

        Returns:
        --------
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
