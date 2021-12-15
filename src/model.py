import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.nn.modules.container import ModuleList
from src.hgru_cell import hConvGRUCell
from src.illusory_cell import IllusConvCell
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

class PredNetVGG(nn.Module):
    def __init__(
        self, model_name, n_classes, n_layers, img_layers,
        seg_layers, bu_channels, td_channels, dropout_rates,
        do_untouched_bu, do_time_aligned, do_prediction, do_segmentation) -> None:
        super(PredNetVGG, self).__init__()

        # Model parameters
        self.model_name = model_name
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.img_layers = img_layers
        self.seg_layers = seg_layers
        self.bu_channels = bu_channels
        self.td_channels = td_channels
        self.dropout_rates = dropout_rates
        self.do_time_aligned = do_time_aligned
        self.do_prediction = do_prediction
        self.do_segmentation = do_segmentation
        self.do_untouched_bu = do_untouched_bu

        # Model states
        self.E_state = [None for _ in range(n_layers)]
        self.R_state = [None for _ in range(n_layers)]
        self.I1_state = [None for _ in range(n_layers)]
        self.I2_state = [None for _ in range(n_layers)]

        # Bottom-up (bu) and lateral (la) connections
        bu_conv = []
        for l in range(self.n_layers):
            in_channels = 3 if l == 0 else (1 + (not do_untouched_bu)) * bu_channels[l - 1]  # ICI MAY BE 4!!!!!
            bu_conv.append(nn.Conv2d(in_channels, bu_channels[l], kernel_size=3, padding=1))
        self.bu_conv = nn.ModuleList(bu_conv)
        self.bu_drop = nn.ModuleList([nn.Dropout(p=r) for r in dropout_rates])
        self.bu_attn = nn.ModuleList([SpatialAttention() for _ in range(self.n_layers)])
        self.la_conv = nn.ModuleList([nn.Conv2d(
            td_channels[l], bu_channels[l], kernel_size=1, padding=0) for l in range(n_layers)])
        
        # Top-down connections (td)
        td_conv = []
        for l in range(self.n_layers):
            in_channels = 2 * bu_channels[l]  # 2 * bu_channels[l] + (0 if l == n_layers - 1 else td_channels[l + 1])
            td_conv.append(hConvGRUCell(in_channels, td_channels[l], kernel_size=3))
        self.td_conv = nn.ModuleList(td_conv)
        self.td_upsample = nn.ModuleList([nn.Upsample(scale_factor=2) for l in range(n_layers - 1)])
        self.td_attn = nn.ModuleList([None] + [ChannelAttention(td_channels[l]) for l in range(1, n_layers)])
        
        # Image prediction
        if self.do_prediction:
            self.img_decoder = Decoder_2D(img_layers, td_channels, 3, nn.Hardtanh(min_val=0.0, max_val=1.0))  # ICI MAY BE 4!!!!

        # Image segmentation
        if self.do_segmentation:
            self.seg_decoder = Decoder_2D(seg_layers, td_channels, n_classes, nn.Sigmoid())

        # Put model on gpu and create folder for the model
        self.to('cuda')
        os.makedirs(f'./ckpt/{model_name}/', exist_ok=True)

    def forward(self, A, frame_idx):
        
        # Initialize outputs of this step, as well as internal states, if necessary
        batch_dims = A.size()
        batch_size, _, h, w = batch_dims
        E_pile, R_pile = [None] * self.n_layers, [None] * self.n_layers
        I1_pile, I2_pile = [None] * self.n_layers, [None] * self.n_layers
        if frame_idx == 0:
            for l in range(self.n_layers):
                self.E_state[l] = torch.zeros(batch_size, 2 * self.bu_channels[l], h, w).cuda()
                self.R_state[l] = torch.zeros(batch_size, self.td_channels[l], h, w).cuda()
                self.I1_state[l] = torch.zeros(batch_size, self.td_channels[l], h, w).cuda()
                self.I2_state[l] = torch.zeros(batch_size, self.td_channels[l], h, w).cuda()
                h, w = h // 2, w // 2

        # Top-down pass
        for l in reversed(range(self.n_layers)):
            R = self.R_state[l]
            E = self.E_state[l]
            if l < self.n_layers - 1:
                E = self.bu_attn[l](E) * E  # bottom-up (spatial) attention
                td_input = self.td_upsample[l](self.R_state[l + 1])
                td_input = self.td_attn[l + 1](td_input) * td_input  # top-down (feature) attention
                E = E + td_input  # torch.cat((E, td_input), dim=1)
            R_pile[l] = self.td_conv[l](E, R)

        # Bottom-up pass
        for l in range(self.n_layers):
            A_hat = F.relu(self.la_conv[l](self.R_state[l]))  # post-synaptic activity of representation prediction add ReLU???
            A = self.bu_conv[l](self.bu_drop[l](A))  # presynaptic activity of bottom-up input
            error = F.relu(torch.cat((A - A_hat, A_hat - A), dim=1))  # post-synaptic activity of A (error goes up)
            E_pile[l] = error  # stored for later: used in top-down pass
            A = F.max_pool2d(A if self.do_untouched_bu else error, kernel_size=2, stride=2)  # A update for next bu-layer

        # Image prediction
        if self.do_prediction:
            img_prediction = self.img_decoder(R_pile)
        else:
            img_prediction = torch.zeros(batch_dims).cuda()

        # Image segmentation
        if self.do_segmentation:
            img_segmentation = self.seg_decoder(R_pile)
        else:
            segm_dims = (batch_size, self.n_classes) + batch_dims[2:]
            img_segmentation = torch.zeros(segm_dims).cuda()
            
        # Update the states of the network
        self.E_state = E_pile
        self.R_state = R_pile
        self.I1_state = I1_pile
        self.I2_state = I2_pile

        # Return the states to the computer
        return E_pile, img_prediction, img_segmentation

    def save_model(self, optimizer, scheduler, train_losses, valid_losses):

        last_epoch = scheduler.last_epoch
        torch.save({
            'model_name': self.model_name,
            'n_classes': self.n_classes,
            'n_layers': self.n_layers,
            'img_layers': self.img_layers,
            'seg_layers': self.seg_layers,
            'bu_channels': self.bu_channels,
            'td_channels': self.td_channels,
            'dropout_rates': self.dropout_rates,
            'do_untouched_bu': self.do_untouched_bu,
            'do_time_aligned': self.do_time_aligned,
            'do_prediction': self.do_prediction,
            'do_segmentation': self.do_segmentation,
            'model_params': self.state_dict(),
            'optimizer_params': optimizer.state_dict(),
            'scheduler_params': scheduler.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses},
            f'./ckpt/{self.model_name}/ckpt_{last_epoch:03}.pt')
        print('SAVED')
        train_losses = [l if l < 10 * sum(train_losses) / len(train_losses) else 0.0 for l in train_losses]
        valid_losses = [l if l < 10 * sum(valid_losses) / len(valid_losses) else 0.0 for l in valid_losses]
        train_axis = list(np.arange(0, last_epoch, last_epoch / len(train_losses)))[:len(train_losses)]
        valid_axis = list(np.arange(0, last_epoch, last_epoch / len(valid_losses)))[:len(valid_losses)]
        plt.plot(train_axis, train_losses, label='train')
        plt.plot(valid_axis, valid_losses, label='valid')
        plt.legend()
        plt.savefig(f'./ckpt/{self.model_name}/loss_plot.png')
        plt.close()

    @classmethod
    def load_model(cls, model_name, lr_params, n_epochs_run, epoch_to_load=None):

        ckpt_dir = f'./ckpt/{model_name}/'
        list_dir = [c for c in os.listdir(ckpt_dir) if '.pt' in c]
        ckpt_path = list_dir[-1]  # take last checkpoint (default)
        for ckpt in list_dir:
            if str(epoch_to_load) in ckpt.split('_')[-1]:
                ckpt_path = ckpt
        save = torch.load(ckpt_dir + ckpt_path)
        model = cls(
            model_name=model_name,
            n_classes=save['n_classes'],
            n_layers=save['n_layers'],
            img_layers=save['img_layers'],
            seg_layers=save['seg_layers'],
            bu_channels=save['bu_channels'],
            td_channels=save['td_channels'],
            dropout_rates=save['dropout_rates'],
            do_untouched_bu=save['do_untouched_bu'],
            do_time_aligned=save['do_time_aligned'],
            do_prediction=save['do_prediction'],
            do_segmentation=save['do_segmentation'])
        model.load_state_dict(save['model_params'])
        scheduler_type, learning_rate, lr_decay_time, lr_decay_rate, betas, first_cycle_steps,\
            cycle_mult, max_lr, min_lr, warmup_steps, gamma, betas = lr_params
        optimizer = torch.optim.AdamW(
            params=model.parameters(), lr=learning_rate, betas=betas)
        if scheduler_type == 'multistep':    
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                range(lr_decay_time, (n_epochs_run + 1) * 10, lr_decay_time),
                gamma=lr_decay_rate)
        elif scheduler_type == 'cosannealwarmuprestart':
            scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=first_cycle_steps,
                                                  cycle_mult=cycle_mult, max_lr=max_lr, min_lr=min_lr,
                                                  warmup_steps=warmup_steps, gamma=gamma)
        optimizer.load_state_dict(save['optimizer_params'])
        scheduler.load_state_dict(save['scheduler_params'])
        valid_losses = save['valid_losses']
        train_losses = save['train_losses']
        return model, optimizer, scheduler, train_losses, valid_losses


# Taken from https://openaccess.thecvf.com/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html
class Decoder_2D(nn.Module):
    def __init__(self, decoder_layers, input_channels, n_output_channels, output_fn):
        super(Decoder_2D, self).__init__()
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
        