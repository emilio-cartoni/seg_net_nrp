import torch
import torch.utils.data as data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import h5py
import albumentations as A
from albumentations.pytorch import ToTensorV2
# VGG_MEAN = np.array([0.485, 0.456, 0.406])
# VGG_STD = np.array([0.229, 0.224, 0.225])
VGG_MEAN = np.array([0.0, 0.0, 0.0])
VGG_STD = np.array([1.0, 1.0, 1.0])

class H5Dataset_Seg(data.Dataset):

  def __init__(self, h5_path, from_to, augmentation, remove_ground, speedup_factor, mode):

    # Parameters pre-initialization
    super(H5Dataset_Seg, self).__init__()
    self.h5_path = h5_path
    self.from_to = from_to
    self.augmentation = augmentation
    self.remove_ground = remove_ground
    self.speedup_factor = speedup_factor
    self.do_saliency = h5_path.split('.h5')[0][-1] == 'S'  # code for saliency
    self.img_samples = None
    self.lbl_segment = None
    self.saliences = None
    with h5py.File(h5_path, 'r') as f:
      n_samples, self.n_frames, height, width = f['rgb_samples'].shape[:-1]
      try:
        self.dataset_length = from_to[1] - from_to[0]
      except TypeError:
        self.dataset_length = n_samples - from_to[0]

    # Data augmentation initialization
    if augmentation:
      if mode == 'train':  # could be more transforms here
        self.transform = A.Compose([
          A.Normalize(mean=VGG_MEAN, std=VGG_STD),
          ToTensorV2()],
          additional_targets={f'image{t}': 'image' for t in range(1, 20)})
      elif mode == 'valid':  # just for VGG
        self.transform = A.Compose([
          A.Normalize(mean=VGG_MEAN, std=VGG_STD),
          ToTensorV2()],
          additional_targets={f'image{t}': 'image' for t in range(1, 20)})

  def __getitem__(self, index):

    # Initialize datasets
    index += self.from_to[0]
    if self.img_samples is None:
      self.img_samples = h5py.File(self.h5_path, 'r')['rgb_samples']
      self.lbl_segment = h5py.File(self.h5_path, 'r')['lbl_segments']
      if self.do_saliency:
        self.saliences = h5py.File(self.h5_path, 'r')['lbl_saliences']

    # Data augmentation
    if self.augmentation:
      samples = np.array(self.img_samples[index])
      lbl_segm = np.array(self.lbl_segment[index])
      lbl_segm = [lbl_segm[t] for t in range(self.n_frames)]
      if self.do_saliency:
        saliency = np.array(self.saliences[index])
        saliency = [saliency[t] for t in range(self.n_frames)]
      else:
        saliency = []
      sample0 = samples[0]
      sampleT = {f'image{t}': samples[t] for t in range(1, self.n_frames)}
      all_lbl = lbl_segm + saliency
      augment = self.transform(image=sample0, masks=all_lbl, **sampleT)
      samples = [augment['image']] + [augment[f'image{t}'] for t in range(1, self.n_frames)]
      samples = torch.stack(samples, dim=3)
      lbl_segm = torch.from_numpy(np.array(augment['masks'][:self.n_frames])).long()
      lbl_segm = [F.one_hot(lbl_segm[t]) for t in range(self.n_frames)]
      lbl_segm = torch.stack(lbl_segm, dim=3).permute((2, 0, 1, 3)).float()
      if self.do_saliency:
        saliency = np.array(augment['masks'][self.n_frames:]).transpose((1, 2, 0))
        saliency = np.expand_dims(saliency.astype('float') / 255.0, axis=0)  # still np.array here

    # No augmentation (e.g. for testing)
    else:
      samples = torch.from_numpy(np.array(self.img_samples[index]).transpose((-1, 1, 2, 0))) / 255.0
      lbl_segm = torch.from_numpy(np.array(self.lbl_segment[index]).transpose((1, 2, 0)))
      lbl_segm = F.one_hot(lbl_segm.long()).permute(-1, 0, 1, 2).float()
      if self.do_saliency:
        saliency = np.array(self.saliences[index]).transpose((1, 2, 0))
        saliency = np.expand_dims(saliency.astype('float') / 255.0, axis=0)
      else:
        saliency = []

    # Additional modifications to the data
    if self.speedup_factor > 1:
      zero_frame = np.random.randint(self.speedup_factor)
      samples = samples[..., zero_frame::self.speedup_factor]
      lbl_segm = lbl_segm[..., zero_frame::self.speedup_factor]
      if self.do_saliency:
        saliency = saliency[..., zero_frame::self.speedup_factor]
    if self.remove_ground:
      lbl_segm = lbl_segm[1:]

    # Return the sample sequence to the computer
    return samples.to(device='cuda'), lbl_segm.to(device='cuda')  #, saliency

  def __len__(self):
    return self.dataset_length


def get_datasets_seg(
  dataset_path, tr_ratio, n_samples, batch_size_train, batch_size_valid,
  augmentation=False, remove_ground=True, speedup_factor=1, mode=None):
  
  # Training dataloader
  if mode != 'test':
    train_bounds = (0, int(n_samples * tr_ratio))
    train_dataset = H5Dataset_Seg(
      dataset_path, train_bounds, True, remove_ground, speedup_factor, 'train')
    train_dataloader = data.DataLoader(
      train_dataset, batch_size=batch_size_train, shuffle=True,
      sampler=None, batch_sampler=None, num_workers=0, collate_fn=None,
      pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
      prefetch_factor=2, persistent_workers=False)
  else:
    train_dataloader = None

  # Validation dataloader
  valid_bounds = (int(n_samples * tr_ratio), None)  # None means very last one
  valid_dataset = H5Dataset_Seg(
    dataset_path, valid_bounds, augmentation, remove_ground, speedup_factor, 'valid')
  valid_dataloader = data.DataLoader(
    valid_dataset, batch_size=batch_size_valid, shuffle=True,
    sampler=None, batch_sampler=None, num_workers=0, collate_fn=None,
    pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
    prefetch_factor=2, persistent_workers=False)

  # Return the dataloaders to the computer
  return train_dataloader, valid_dataloader
