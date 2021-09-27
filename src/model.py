import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from src.hgru_cell import hConvGRUCell

# Attention module taken from:
# https://github.com/openai/CLIP/blob/04f4dc2ca1ed0acc9893bd1a3b526a7e02c4bb10/clip/model.py#L56


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
    pr_layers, sg_layers, td_channels, dropout_rates,
    do_time_aligned, do_untouched_bu, do_train_vgg,
    do_bens_idea, do_saliency, do_prediction, do_segmentation) -> None:
    super(PredNetVGG, self).__init__()

    # Model parameters
    self.model_name = model_name
    self.n_classes = n_classes
    self.n_layers = n_layers
    self.pr_layers = pr_layers
    self.sg_layers = sg_layers
    self.bu_channels = [64, 128, 256, 512, 512][:n_layers]  # vgg channels
    self.td_channels = td_channels[:n_layers] + (0,)
    self.do_time_aligned = do_time_aligned
    self.do_untouched_bu = do_untouched_bu
    self.do_bens_idea = do_bens_idea
    self.do_saliency = do_saliency
    self.do_prediction = do_prediction
    self.do_segmentation = do_segmentation

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
      bu_conv.append(vgg.features[vgg_indexes[vgg_type][l][0]:vgg_indexes[vgg_type][l][1]])
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
        nn.Conv2d(in_channels=inn, out_channels=out, kernel_size=3, padding=1),
        nn.ReLU()))
    self.la_conv = nn.ModuleList(la_conv)

    # Top-down connections (td)
    td_conv, td_norm, td_attn = [], [], []
    spatial_dims = [64, 32, 16, 8, 4, 2, 1]
    for l in range(self.n_layers):
      inn, out = self.bu_channels[l] + self.td_channels[l + 1], self.td_channels[l]
      td_conv.append(hConvGRUCell(inn, out, kernel_size=3))
      # td_conv.append(nn.Conv2d(inn, out, kernel_size=3, padding=1))
      td_norm.append(nn.GroupNorm(out, out))
      td_attn.append(AttentionPool2d(
        spatial_dim=spatial_dims[l], embed_dim=inn, num_heads=4, output_dim=out))
    self.td_conv = nn.ModuleList(td_conv)
    self.td_upsp = nn.Upsample(scale_factor=2)
    self.td_attn = nn.ModuleList(td_attn)
    self.td_norm = nn.ModuleList(td_norm)

    # Future frame prediction (pr)
    if self.do_prediction:
      pr_upsp = []
      for l in range(self.n_layers):
        if 0 < l <= max(self.pr_layers):
          inn, out = self.td_channels[l], self.td_channels[l - 1]
          pr_upsp.append(nn.Sequential(
            nn.GroupNorm(inn, inn),
            nn.ConvTranspose2d(in_channels=inn, out_channels=out,
              kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()))  # [int(l == max(self.pr_layers)):])
      self.pr_upsp = nn.ModuleList([None] + pr_upsp)  # None for indexing convenience
      self.pr_conv = nn.Sequential(
        nn.GroupNorm(self.td_channels[0], self.td_channels[0]),
        nn.Conv2d(self.td_channels[0], 3, kernel_size=1),
        nn.Hardtanh(min_val=0.0, max_val=1.0))
      self.register_parameter(name='pr_prod', param=torch.nn.Parameter(torch.tensor(
          [1.0 / max(self.pr_layers)] * (1 + max(self.pr_layers)))))

    # Segmentation prediction (sg)
    if self.do_segmentation:
      sg_upsp = []
      for l in range(self.n_layers):
        if 0 < l <= max(self.sg_layers):
          inn, out = self.td_channels[l], self.td_channels[l - 1]
          sg_upsp.append(nn.Sequential(
            nn.GroupNorm(inn, inn),
            nn.ConvTranspose2d(in_channels=inn, out_channels=out,
              kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()))  # [int(l == max(self.sg_layers)):])
      self.sg_upsp = nn.ModuleList([None] + sg_upsp)  # None for indexing convenience
      self.sg_conv = nn.Sequential(
        nn.GroupNorm(self.td_channels[0], self.td_channels[0]),
        nn.Conv2d(self.td_channels[0], self.n_classes, kernel_size=1),
        nn.Sigmoid())
      self.register_parameter(name='sg_prod', param=torch.nn.Parameter(torch.tensor(
          [1.0 / max(self.sg_layers)] * (1 + max(self.sg_layers)))))
      
    # # Future frame prediction (pr)
    # if self.do_prediction:
    #   self.pr_relu = nn.ReLU(inplace=True)
    #   self.pr_deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    #   self.pr_bn1 = nn.GroupNorm(128, 128)
    #   self.pr_deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    #   self.pr_bn2 = nn.GroupNorm(64, 64)
    #   self.pr_deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    #   self.pr_bn3 = nn.GroupNorm(32, 32)
    #   self.pr_classifier = nn.Conv2d(32, 3, kernel_size=1)
    #   self.pr_hardtanh = nn.Hardtanh(min_val=0.0, max_val=1.0, inplace=False)

    # # Segmentation prediction (sg)
    # if self.do_segmentation:
    #   self.sg_relu = nn.ReLU(inplace=True)
    #   self.sg_deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    #   self.sg_bn1 = nn.GroupNorm(128, 128)
    #   self.sg_deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    #   self.sg_bn2 = nn.GroupNorm(64, 64)
    #   self.sg_deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    #   self.sg_bn3 = nn.GroupNorm(32, 32)
    #   self.sg_classifier = nn.Conv2d(32, self.n_classes, kernel_size=1)
    #   self.sg_sigmoid = nn.Sigmoid()

    # Put model on gpu
    self.to('cuda')

  def forward(self, A, frame_idx):
    
    # Initialize outputs of this step, as well as internal states, if necessary
    batch_dims = A.size()
    batch_size, n_channels, h, w = batch_dims
    A_pile, E_pile, error_pile, R_pile = [None]*self.n_layers, [None]*self.n_layers, [None]*self.n_layers, [None]*self.n_layers
    if frame_idx == 0:  # replace by if self.E_state[l] == None? network might freak out every new sequence
      for l in range(self.n_layers):
        self.A_state[l] = torch.zeros(batch_size, self.bu_channels[l], h, w).cuda()
        self.E_state[l] = torch.zeros(batch_size, self.bu_channels[l], h, w).cuda()
        self.R_state[l] = torch.zeros(batch_size, self.td_channels[l], h, w).cuda()
        h, w = h // 2, w // 2
      
    # Top-down pass
    for l in reversed(range(self.n_layers)):
      R = self.R_state[l]
      E = self.E_state[l]
      if l != self.n_layers - 1:
        E = torch.cat((E, self.td_upsp(self.R_state[l + 1])), dim=1)
      E = self.td_attn[l](E)
      R_pile[l] = self.td_norm[l](self.td_conv[l](E, R))

    # Bottom-up pass
    for l in range(self.n_layers):
      A = self.bu_drop[l](self.bu_conv[l](A))
      A_hat = self.la_conv[l](self.R_state[l])
      error = torch.abs(A - A_hat)
      if self.do_saliency:
        E_pile[l] = A * (1.0 + error / error.sum())
      else:
        E_pile[l] = error
      error_pile[l] = error
      A_pile[l] = A
      if l < self.n_layers - 1:
        if self.do_untouched_bu:
          A = self.bu_pool(self.A_state[l])
        else:
          A = self.bu_pool(self.E_state[l])

    # Future frame prediction
    if self.do_prediction:
      P = R_pile[max(self.pr_layers)] * self.pr_prod[-1]
      P = self.pr_upsp[-1](P) if max(self.pr_layers) > 0 else self.pr_conv(P)
      for l in reversed(range(max(self.pr_layers))):
        if l in self.pr_layers:
          P = P + R_pile[l] * self.pr_prod[l]
        P = self.pr_upsp[l](P) if l > 0 else self.pr_conv(P)
    else:
      P = torch.zeros(batch_dims).cuda()

    # Segmentation
    if self.do_segmentation:
      S = R_pile[max(self.sg_layers)] * self.sg_prod[-1]
      S = self.sg_upsp[-1](S) if max(self.sg_layers) > 0 else self.sg_conv(S)
      for l in reversed(range(max(self.sg_layers))):
        if l in self.sg_layers:
          S = S + R_pile[l] * self.sg_prod[l]
        S = self.sg_upsp[l](S) if l > 0 else self.sg_conv(S)
    else:
      S = torch.zeros((batch_size, self.n_classes) + batch_dims[2:]).cuda()

    # # Future frame prediction
    # if self.do_prediction:
    #   P_input = [None for _ in self.pr_layers]
    #   for i, l in enumerate(self.pr_layers):
    #     P_input[i] = R_pile[l]
    #   P = self.pr_relu(self.pr_deconv1(P_input[-1]))  # size=(N, 512, x.H/16, x.W/16)
    #   P = self.pr_bn1(P + P_input[-2])  # element-wise add, size=(N, 512, x.H/16, x.W/16)
    #   P = self.pr_relu(self.pr_deconv2(P))  # size=(N, 256, x.H/8, x.W/8)
    #   P = self.pr_bn2(P + P_input[-3])  # element-wise add, size=(N, 256, x.H/8, x.W/8)
    #   P = self.pr_relu(self.pr_deconv3(P))
    #   P = self.pr_bn3(P + P_input[-4])  # size=(N, 128, x.H/4, x.W/4)
    #   P = self.pr_classifier(P)  # size=(N, n_class, x.H/1, x.W/1)
    #   P = self.pr_hardtanh(P)
    # else:
    #   P = torch.zeros(batch_dims).cuda()

    # # Segmentation
    # if self.do_segmentation:
    #   S_input = [None for _ in self.sg_layers]
    #   for i, l in enumerate(self.sg_layers):
    #     S_input[i] = R_pile[l]
    #   S = self.sg_relu(self.sg_deconv1(S_input[-1]))  # size=(N, 512, x.H/16, x.W/16)
    #   S = self.sg_bn1(S + S_input[-2])  # element-wise add, size=(N, 512, x.H/16, x.W/16)
    #   S = self.sg_relu(self.sg_deconv2(S))  # size=(N, 256, x.H/8, x.W/8)
    #   S = self.sg_bn2(S + S_input[-3])  # element-wise add, size=(N, 256, x.H/8, x.W/8)
    #   S = self.sg_relu(self.sg_deconv3(S))
    #   S = self.sg_bn3(S + S_input[-4])  # size=(N, 128, x.H/4, x.W/4)
    #   S = self.sg_classifier(S)  # size=(N, n_class, x.H/1, x.W/1)
    #   S = self.sg_sigmoid(S)
    # else:
    #   S = torch.zeros((batch_size, self.n_classes) + batch_dims[2:]).cuda()

    # Update the states of the network
    self.A_state = A_pile
    self.E_state = E_pile
    self.R_state = R_pile

    # Return the states to the computer
    return error_pile, P, S

  def save_model(self, optimizer, scheduler, train_losses, valid_losses):

    last_epoch = scheduler.last_epoch
    torch.save({
      'model_name': self.model_name,
      'vgg_type': self.vgg_type,
      'n_classes': self.n_classes,
      'n_layers': self.n_layers,
      'pr_layers': self.pr_layers,
      'sg_layers': self.sg_layers,
      'td_channels': self.td_channels,
      'do_time_aligned': self.do_time_aligned,
      'do_untouched_bu': self.do_untouched_bu,
      'do_bens_idea': self.do_bens_idea,
      'do_saliency': self.do_saliency,
      'do_prediction': self.do_prediction,
      'do_segmentation': self.do_segmentation,
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
      td_channels=save['td_channels'],
      do_time_aligned=save['do_time_aligned'],
      do_untouched_bu=save['do_untouched_bu'],
      do_bens_idea=save['do_bens_idea'],
      do_saliency=save['do_saliency'],
      do_prediction=save['do_prediction'],
      do_segmentation=save['do_segmentation'])
    model.load_state_dict(save['model_params'])
    optimizer = save['optimizer']
    scheduler = save['scheduler']
    optimizer.load_state_dict(save['optimizer_params'])
    scheduler.load_state_dict(save['scheduler_params'])
    valid_losses = save['valid_losses']
    train_losses = save['train_losses']
    return model, optimizer, scheduler, train_losses, valid_losses


# Attention module taken from:
# https://github.com/openai/CLIP/blob/04f4dc2ca1ed0acc9893bd1a3b526a7e02c4bb10/clip/model.py#L56
class AttentionPool2d(nn.Module):
    def __init__(self, spatial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spatial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]