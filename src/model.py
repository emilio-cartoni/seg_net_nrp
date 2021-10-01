import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from src.hgru_cell import hConvGRUCell

# Attention modules taken from:
# https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py

vgg_indexes = {
    'vgg13': [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24)],
    'vgg16': [(0, 4), (5, 9), (10, 16), (17, 23), (24, 30)],
    'vgg19': [(0, 4), (5, 9), (10, 18), (19, 27), (28, 36)],
    'vgg13_bn': [(0, 6), (7, 13), (14, 20), (21, 27), (28, 34)],
    'vgg16_bn': [(0, 6), (7, 13), (14, 23), (24, 33), (34, 43)],
    'vgg19_bn': [(0, 6), (7, 13), (14, 26), (27, 39), (40, 51)]}


class PredNetVGG(nn.Module):
    def __init__(
        self, model_name, vgg_type, n_classes, n_layers,
        pr_layers, sg_layers, saccade_layers, td_channels, dropout_rates,
        do_time_aligned, do_untouched_bu, do_train_vgg,
        do_prediction, do_segmentation, do_saccades) -> None:
        super(PredNetVGG, self).__init__()

        # Model parameters
        self.model_name = model_name
        self.vgg_type = vgg_type
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.pr_layers = pr_layers
        self.sg_layers = sg_layers
        self.saccade_layers = saccade_layers
        self.bu_channels = [64, 128, 256, 512, 512][:n_layers]  # vgg channels
        self.td_channels = td_channels[:n_layers] + (0,)
        self.dropout_rates = dropout_rates
        self.do_time_aligned = do_time_aligned
        self.do_untouched_bu = do_untouched_bu
        self.do_train_vgg = do_train_vgg
        self.do_prediction = do_prediction
        self.do_segmentation = do_segmentation
        self.do_saccades = do_saccades

        # Model directory
        model_path = f'./ckpt/{model_name}/'
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        # Model states
        self.A_state = [None for _ in range(n_layers)]
        self.E_state = [None for _ in range(n_layers)]
        self.R_state = [None for _ in range(n_layers)]

        # Bottom-up connections (bu)
        vgg = torch.hub.load('pytorch/vision:v0.10.0', vgg_type, pretrained=True)
        bu_conv = []
        for l in range(self.n_layers):
            bu_conv.append(
                vgg.features[vgg_indexes[vgg_type][l][0]:vgg_indexes[vgg_type][l][1]])
        self.bu_conv = nn.ModuleList(bu_conv)
        if not do_train_vgg:
            for param in self.bu_conv.parameters():
                param.requires_grad = False
        self.bu_pool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.bu_drop = nn.ModuleList([nn.Dropout2d(p=r) for r in dropout_rates])
        
        # Lateral connections (la)
        la_conv = []
        for l in range(self.n_layers):
            inn, out = self.td_channels[l], self.bu_channels[l]
            la_conv.append(nn.Sequential(
                nn.Conv2d(in_channels=inn, out_channels=out, kernel_size=1, padding=0),
                nn.ReLU()))
        self.la_conv = nn.ModuleList(la_conv)

        # Top-down connections (td)
        td_conv, td_attn_layer, td_attn_channel, td_attn_spatial = [], [], [], []
        for l in range(self.n_layers):
            inn, out = self.bu_channels[l] + self.td_channels[l + 1], self.td_channels[l]
            td_attn_channel.append(ChannelAttention(inn))
            td_attn_spatial.append(SpatialAttention())
            td_attn_layer.append(nn.Sequential(ChannelAttention(inn), SpatialAttention()))
            td_conv.append(hConvGRUCell(inn, out, kernel_size=5))
            # td_conv.append(nn.Conv2d(inn, out, kernel_size=3, padding=1))
        self.td_upsp = nn.Upsample(scale_factor=2)
        self.td_attn_layer = nn.ModuleList(td_attn_layer)
        self.td_attn_channel = nn.ModuleList(td_attn_channel)
        self.td_attn_spatial = nn.ModuleList(td_attn_spatial)
        self.td_conv = nn.ModuleList(td_conv)

        # Future frame prediction
        if self.do_prediction:
            self.img_decoder = Decoder(pr_layers, td_channels, 3, nn.Hardtanh(min_val=0.0, max_val=1.0))

        # Segmentation prediction
        if self.do_segmentation:
            self.seg_decoder = Decoder(sg_layers, td_channels, n_classes, nn.Sigmoid())

        # TODO: # FOR NOW THESE WEIGHTS ARE ACTUALLY NOT TRAINED!!!!!
        # SO THE SOLUTION IS TO MAKE A HIERARCHICAL DECISION BASED ON ALL SPATIAL ATTENTION LAYERS
        # SPECIFICALLY THE HIGHEST LVL LAYER CONSTRAINS THE NEXT LAYER DOWN, ETC, UNTIL THE BOTTOM IS REACHED
        # Saccade generation
        if self.do_saccades:
            self.saccade_decoder = Decoder(saccade_layers, td_channels, 1, nn.Sigmoid())

        # Put model on gpu
        self.to('cuda')

    def forward(self, A, frame_idx):
        
        # Initialize outputs of this step, as well as internal states, if necessary
        batch_dims = A.size()
        batch_size, n_channels, h, w = batch_dims
        A_pile, E_pile, R_pile = [None]*self.n_layers, [None]*self.n_layers, [None]*self.n_layers
        if frame_idx == 0:
            for l in range(self.n_layers):
                self.A_state[l] = torch.zeros(batch_size, self.bu_channels[l], h, w).cuda()
                self.E_state[l] = torch.zeros(batch_size, self.bu_channels[l], h, w).cuda()
                self.R_state[l] = torch.zeros(batch_size, self.td_channels[l], h, w).cuda()
                h, w = h // 2, w // 2
        
        # Bottom-up pass
        for l in range(self.n_layers):
            A = self.bu_drop[l](self.bu_conv[l](A))
            A_hat = self.la_conv[l](self.R_state[l])
            E_pile[l] = torch.abs(A - A_hat)
            A_pile[l] = A
            if l < self.n_layers - 1:
                if self.do_untouched_bu:
                    A = self.bu_pool(self.A_state[l])
                else:
                    A = self.bu_pool(self.E_state[l])

        # Top-down pass
        for l in reversed(range(self.n_layers)):
            R = self.R_state[l]
            E = self.E_state[l]
            if l != self.n_layers - 1:
                E = torch.cat((E, self.td_upsp(self.R_state[l + 1])), dim=1)
            # layer_attention = self.td_attn_layer[l](E)
            channel_attention = self.td_attn_channel[l](E)
            spatial_attention = self.td_attn_spatial[l](E)
            # E = layer_attention * E
            E = channel_attention * E
            E = spatial_attention * E
            R_pile[l] = self.td_conv[l](E, R)
            # spatial_attention_pile.append(spatial_attention)

        # Future frame prediction
        if self.do_prediction:
            P = self.img_decoder(R_pile)
        else:
            P = torch.zeros(batch_dims).cuda()

        # Segmentation
        if self.do_segmentation:
            S = self.seg_decoder(R_pile)
        else:
            S = torch.zeros((batch_size, self.n_classes) + batch_dims[2:]).cuda()
        
        # Saccades
        saccade_loc = None
        if self.do_saccades:
            # INSTEAD OF THIS, WE SHOULD USE SPATIAL_ATTENTION_PILE TO DECODE POSITION
            saliency_map = self.saccade_decoder(R_pile)  # (b, c, h, w)
            saliency_argmax = torch.argmax(saliency_map)
            saccade_row = saliency_argmax // saliency_map.shape[-1]
            saccade_col = saliency_argmax % saliency_map.shape[-1]
            saccade_loc = (saccade_row.cpu().item(), saccade_col.cpu().item())

        # Update the states of the network
        self.A_state = A_pile
        self.E_state = E_pile
        self.R_state = R_pile

        # Return the states to the computer
        return E_pile, P, S, saccade_loc

    def save_model(self, optimizer, scheduler, train_losses, valid_losses):

        last_epoch = scheduler.last_epoch
        torch.save({
            'model_name': self.model_name,
            'vgg_type': self.vgg_type,
            'n_classes': self.n_classes,
            'n_layers': self.n_layers,
            'pr_layers': self.pr_layers,
            'sg_layers': self.sg_layers,
            'saccade_layers': self.saccade_layers,
            'td_channels': self.td_channels,
            'dropout_rates': self.dropout_rates,
            'do_time_aligned': self.do_time_aligned,
            'do_untouched_bu': self.do_untouched_bu,
            'do_train_vgg': self.do_train_vgg,
            'do_prediction': self.do_prediction,
            'do_segmentation': self.do_segmentation,
            'do_saccades': self.do_saccades,
            'model_params': self.state_dict(),
            'optimizer': optimizer,
            'scheduler': scheduler,
            'optimizer_params': optimizer.state_dict(),
            'scheduler_params': scheduler.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses},
            f'./ckpt/{self.model_name}/ckpt_{last_epoch:02}.pt')
        print('SAVED')
        plt.plot(list(range(last_epoch)), valid_losses, label='valid')
        plt.plot(list(range(last_epoch)), train_losses, label='train')
        plt.legend()
        plt.savefig(f'./ckpt/{self.model_name}/loss_plot.png')
        plt.close()

    @classmethod
    def load_model(cls, model_name, epoch_to_load=None):

        ckpt_dir = f'./ckpt/{model_name}/'
        list_dir = [c for c in os.listdir(ckpt_dir) if ('decoder' not in c and '.pt' in c)]
        ckpt_path = list_dir[-1]  # take last checkpoint (default)
        for ckpt in list_dir:
            if str(epoch_to_load) in ckpt.split('_')[-1]:
                ckpt_path = ckpt
        save = torch.load(ckpt_dir + ckpt_path)
        model = cls(
            model_name=model_name,
            vgg_type=save['vgg_type'],
            n_classes=save['n_classes'],
            n_layers=save['n_layers'],
            pr_layers=save['pr_layers'],
            sg_layers=save['sg_layers'],
            saccade_layers=save['saccade_layers'],
            td_channels=save['td_channels'],
            dropout_rates=save['dropout_rates'],
            do_time_aligned=save['do_time_aligned'],
            do_untouched_bu=save['do_untouched_bu'],
            do_train_vgg=save['do_train_vgg'],
            do_prediction=save['do_prediction'],
            do_segmentation=save['do_segmentation'],
            do_saccades=save['do_saccades'])
        model.load_state_dict(save['model_params'])
        optimizer = save['optimizer']
        scheduler = save['scheduler']
        optimizer.load_state_dict(save['optimizer_params'])
        scheduler.load_state_dict(save['scheduler_params'])
        valid_losses = save['valid_losses']
        train_losses = save['train_losses']
        return model, optimizer, scheduler, train_losses, valid_losses


class Decoder(nn.Module):  # decode anything from the latent variables of PredNetVGG
    def __init__(self, decoder_layers, input_channels, n_output_channels, output_fn):
        super(Decoder, self).__init__()

        self.decoder_layers = decoder_layers
        decoder_upsp = []
        for l in range(1, max(decoder_layers) + 1):
            inn, out = input_channels[l], input_channels[l - 1]
            decoder_upsp.append(nn.Sequential(
                nn.GroupNorm(inn, inn),
                nn.ConvTranspose2d(in_channels=inn, out_channels=out,
                kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.ReLU()))  # [int(l == max(decoder_layers)):])
        self.decoder_upsp = nn.ModuleList([None] + decoder_upsp)  # None for indexing convenience
        self.decoder_conv = nn.Sequential(
            nn.GroupNorm(input_channels[0], input_channels[0]),
            nn.Conv2d(input_channels[0], n_output_channels, kernel_size=1, bias=False),
            output_fn)
        # self.register_parameter(name='decoder_prod', param=torch.nn.Parameter(
        #     torch.tensor([1.0 / max(decoder_layers)] * (1 + max(decoder_layers)))))
  
    def forward(self, R_pile):

        D = R_pile[max(self.decoder_layers)]  # * self.decoder_prod[-1]
        D = self.decoder_upsp[-1](D) if max(self.decoder_layers) > 0 else self.decoder_conv(D)
        for l in reversed(range(max(self.decoder_layers))):
            if l in self.decoder_layers:
                D = D + R_pile[l]  # * self.decoder_prod[l]
            D = self.decoder_upsp[l](D) if l > 0 else self.decoder_conv(D)
        return D


# Taken from: https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


# Taken from: https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
