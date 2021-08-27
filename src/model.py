import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from src.hconvgru_cell import hConvGRUCell

vgg_indexes = {
  'vgg13': [(0, 4), (5, 9), (10, 14), (15, 19)],
  'vgg16': [(0, 4), (5, 9), (10, 16), (17, 23)],
  'vgg19': [(0, 4), (5, 9), (10, 18), (19, 27)]}


class PredNetVGG(nn.Module):
  def __init__(
    self, model_name, vgg_type, n_classes, n_layers,
    pr_layers, sg_layers, td_channels, dropout_rates,
    do_time_aligned, do_untouched_bu, do_bens_idea) -> None:
    super(PredNetVGG, self).__init__()

    # Model parameters
    self.model_name = model_name
    self.vgg_type = vgg_type
    self.n_classes = n_classes
    self.n_layers = n_layers
    self.pr_layers = pr_layers
    self.sg_layers = sg_layers
    self.bu_channels = [64, 128, 256, 512][:n_layers]  # hardcoded
    self.td_channels = td_channels[:n_layers] + (0,)
    self.dropout_rates = dropout_rates
    self.do_time_aligned = do_time_aligned
    self.do_untouched_bu = do_untouched_bu
    self.do_bens_idea = do_bens_idea

    # Model directory
    model_path = f'./ckpt/{model_name}/'
    if not os.path.exists(model_path):
      os.mkdir(model_path)

    # Bottom-up connections (bu)
    vgg = torch.hub.load('pytorch/vision:v0.10.0', vgg_type, pretrained=True)
    for param in vgg.parameters():
        param.requires_grad = False
    bu_conv = []
    for l in range(self.n_layers):
      bu_conv.append(vgg.features[vgg_indexes[vgg_type][l][0]:vgg_indexes[vgg_type][l][1]])
    self.bu_conv = nn.ModuleList(bu_conv)
    self.bu_pool = nn.MaxPool2d(
      kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    self.bu_drop = nn.ModuleList([nn.Dropout2d(p=r) for r in self.dropout_rates])

    # Lateral connections (la)
    la_conv = []
    for l in range(self.n_layers):
      inn, out = self.td_channels[l], self.bu_channels[l]
      la_conv.append(nn.Sequential(
        nn.Conv2d(in_channels=inn, out_channels=out, kernel_size=3, padding=1),
        nn.ReLU()))
    self.la_conv = nn.ModuleList(la_conv)

    # Top-down connections (td)
    td_hgru = []
    for l in range(self.n_layers):
      if do_bens_idea:  # (td[l + 1] must be equal to bu[l])
        inn, out = self.bu_channels[l], self.td_channels[l]
      else:
        inn, out = self.bu_channels[l] + self.td_channels[l + 1], self.td_channels[l]
      td_hgru.append(hConvGRUCell(input_size=inn, hidden_size=out, kernel_size=3))
    self.td_hgru = nn.ModuleList(td_hgru)
    self.td_upsp = nn.Upsample(scale_factor=2)

    # Future frame prediction (pr)
    pr_upsp = []
    for l in self.pr_layers:
      to_append = [nn.Sequential(
        nn.ConvTranspose2d(  # or nn.Upsample(scale_factor=2)?
          in_channels=self.td_channels[l], out_channels=self.td_channels[l],
          kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.GroupNorm(self.td_channels[l] // 4, self.td_channels[l])) for _ in range(l)]
      pr_upsp.append(nn.Sequential(*to_append))
    self.pr_upsp = nn.ModuleList(pr_upsp)
    pr_channels = sum([self.td_channels[l] for l in self.pr_layers])
    self.pr_conv = nn.Sequential(
      nn.Conv2d(in_channels=pr_channels, out_channels=3, kernel_size=3, padding=1),
      nn.Hardtanh(min_val=0.0, max_val=1.0, inplace=False))  # range for image prediction

    # Segmentation prediction (sg)
    sg_upsp = []
    for l in self.sg_layers:
      to_append = [nn.Sequential(
        nn.ConvTranspose2d(  # or nn.Upsample(scale_factor=2)?
          in_channels=self.td_channels[l], out_channels=self.td_channels[l],
          kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.GroupNorm(self.td_channels[l] // 4, self.td_channels[l])) for _ in range(l)]
      sg_upsp.append(nn.Sequential(*to_append))
    self.sg_upsp = nn.ModuleList(sg_upsp)
    sg_channels = sum([self.td_channels[l] for l in self.sg_layers])
    self.sg_conv = nn.Sequential(
      nn.Conv2d(in_channels=sg_channels, out_channels=self.n_classes, kernel_size=3, padding=1),
      nn.Sigmoid())

    # Put model on gpu
    self.to('cuda')

  def forward(self, input_sequence):
    
    # Initialize inputs, outputs and internal states
    TA = self.do_time_aligned  # for convenience
    batch_size, n_channels, h, w, n_frames = input_sequence.size()
    P_seq = torch.zeros(batch_size, n_channels, h, w, TA + n_frames).cuda()
    S_seq = torch.zeros(batch_size, self.n_classes, h, w, TA + n_frames).cuda()
    E_seq, R_seq = [], []
    for l in range(self.n_layers):
      E_seq.append([None] * (TA + n_frames))  # [n_l, n_t + 1][b, bu_c, h, w]
      R_seq.append([None] * (TA + n_frames))  # [n_l, n_t + 1][b, td_c, h, w]
      for t in range(n_frames):
        if t == 0 or not TA:
          E_seq[l][t] = torch.zeros(batch_size, 1 * self.bu_channels[l], h, w).cuda()
          R_seq[l][t] = torch.zeros(batch_size, 1 * self.td_channels[l], h, w).cuda()
      h, w = h // 2, w // 2

    # Loop through time run the model
    for t in range(TA, TA + n_frames):
      A = input_sequence[..., t - TA]

      # Top-down pass
      for l in reversed(range(self.n_layers)):
        R = R_seq[l][t - TA]
        E = E_seq[l][max(t - 1, 0)]  # not "t-TA" on purpose
        if l != self.n_layers - 1:
          if self.do_bens_idea:
            E = E * self.td_upsp(R_seq[l + 1][t - TA])
          else:
            E = torch.cat((E, self.td_upsp(R_seq[l + 1][t - TA])), dim=1)
        R_seq[l][t] = self.td_hgru[l](E, R, t - TA)

      # Bottom-up pass
      for l in range(self.n_layers):
        A = self.bu_drop[l](self.bu_conv[l](A))
        A_hat = self.la_conv[l](R_seq[l][t - TA])
        # E_seq[l][t] = torch.cat([F.relu(A_hat - A), F.relu(A - A_hat)], 1)
        E_seq[l][t] = torch.abs(A - A_hat)
        if l < self.n_layers - 1:
          pool_input = A if self.do_untouched_bu else E_seq[l][t]
          A = self.bu_pool(pool_input)  # -> E_seq[l+1]

      # Future frame prediction
      P_input = [None for _ in self.pr_layers]
      for i, l in enumerate(self.pr_layers):
        P_input[i] = self.pr_upsp[i](R_seq[l][t - TA])
      P_seq[..., t] = self.pr_conv(torch.cat(P_input, dim=1))

      # Segmentation
      S_input = [None for _ in self.sg_layers]
      for i, l in enumerate(self.sg_layers):
        S_input[i] = self.sg_upsp[i](R_seq[l][t - TA])
      S_seq[..., t] = self.sg_conv(torch.cat(S_input, dim=1))

    # Return the states to the computer
    return [E[TA:] for E in E_seq], P_seq[..., TA:], S_seq[..., TA:]


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
      'dropout_rates': self.dropout_rates,
      'do_time_aligned': self.do_time_aligned,
      'do_untouched_bu': self.do_untouched_bu,
      'do_bens_idea': self.do_bens_idea,
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
      dropout_rates=save['dropout_rates'],
      do_time_aligned=save['do_time_aligned'],
      do_untouched_bu=save['do_untouched_bu'],
      do_bens_idea=save['do_bens_idea'])
    model.load_state_dict(save['model_params'])
    optimizer = save['optimizer']
    scheduler = save['scheduler']
    optimizer.load_state_dict(save['optimizer_params'])
    scheduler.load_state_dict(save['scheduler_params'])
    valid_losses = save['valid_losses']
    train_losses = save['train_losses']
    return model, optimizer, scheduler, train_losses, valid_losses
