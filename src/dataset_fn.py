from os import remove
import torch
import torch.utils.data as data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import h5py
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationDataset(data.Dataset):

    def __init__(self, h5_path, from_to, n_classes, augmentation, remove_ground, speedup_factor):

        # Parameters pre-initialization
        super(SegmentationDataset, self).__init__()
        self.h5_path = h5_path
        self.from_to = from_to
        self.n_classes = n_classes + int(remove_ground)
        self.augmentation = augmentation
        self.remove_ground = remove_ground
        self.speedup_factor = speedup_factor
        self.img_samples = None
        self.lbl_segment = None
        with h5py.File(h5_path, 'r') as f:
            n_samples, self.n_frames, height, width = f['rgb_samples'].shape[:-1]
            try:
                self.dataset_length = from_to[1] - from_to[0]
            except TypeError:
                self.dataset_length = n_samples - from_to[0]

        # Data augmentation initialization
        min_max_h = (int(height * 0.9), height - 1)
        self.transform = A.Compose([
            A.OneOf([
                A.RandomSizedCrop(
                    min_max_height=min_max_h, height=height, width=width, p=0.5),
                A.PadIfNeeded(min_height=height, min_width=width, p=0.5)], p=1),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=(-10, 10)),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            # A.GaussianBlur(),  # keep? for now, no
            # A.RandomScale(scale_limit=(-0.5, -0.5)),
            # A.Resize(128, 128),
            A.RandomCrop(260, 260),
            A.ColorJitter(),
            A.ChannelDropout(),
            A.ChannelShuffle(),
            A.GaussNoise(var_limit=(0.0, 50.0), mean=0.0),
            A.Normalize((0.0,), (1.0,)),
            ToTensorV2()],
            additional_targets={f'image{t}': 'image' for t in range(1, 20)})

    def __getitem__(self, index):

        # Initialize datasets
        index += self.from_to[0]
        if self.img_samples is None:
            self.img_samples = h5py.File(self.h5_path, 'r')['rgb_samples']
            self.lbl_segment = h5py.File(self.h5_path, 'r')['lbl_segments']

        # Data augmentation
        if self.augmentation:
            samples = np.array(self.img_samples[index])
            lbl_segm = np.array(self.lbl_segment[index])
            lbl_segm = [lbl_segm[t] for t in range(self.n_frames)]
            sample0 = samples[0]
            sampleT = {f'image{t}': samples[t] for t in range(1, self.n_frames)}
            augment = self.transform(image=sample0, masks=lbl_segm, **sampleT)
            samples = [augment['image']] + [augment[f'image{t}'] for t in range(1, self.n_frames)]
            samples = torch.stack(samples, dim=3)
            lbl_segm = torch.from_numpy(np.array(augment['masks'][:self.n_frames])).long()
            lbl_segm = [F.one_hot(lbl_segm[t], num_classes=self.n_classes) for t in range(self.n_frames)]
            lbl_segm = torch.stack(lbl_segm, dim=3).permute((2, 0, 1, 3)).float()

        # No augmentation (e.g. for testing)
        else:
            samples = torch.from_numpy(np.array(self.img_samples[index]).transpose((-1, 1, 2, 0))) / 255.0
            lbl_segm = torch.from_numpy(np.array(self.lbl_segment[index]).transpose((1, 2, 0)))
            lbl_segm = F.one_hot(lbl_segm.long(), num_classes=self.n_classes).permute(-1, 0, 1, 2).float()

        # Additional modifications to the data
        if self.speedup_factor > 1:
            zero_frame = np.random.randint(self.speedup_factor)
            samples = samples[..., zero_frame::self.speedup_factor]
            lbl_segm = lbl_segm[..., zero_frame::self.speedup_factor]
        if self.remove_ground:
            lbl_segm = lbl_segm[1:]

        # Return the sample sequence to the computer
        return samples.to(device='cuda'), lbl_segm.to(device='cuda')

    def __len__(self):
        return self.dataset_length


def get_segmentation_dataloaders(dataset_path, tr_ratio, n_samples, batch_size_train, batch_size_valid, n_classes,
                                 augmentation=False, remove_ground=True, speedup_factor=1, mode=None):
    # Training dataloader
    if mode != 'test':
        train_bounds = (0, int(n_samples * tr_ratio))
        train_dataset = SegmentationDataset(dataset_path, train_bounds, n_classes, augmentation, remove_ground, speedup_factor)
        train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    else:
        train_dataloader = None

    # Validation dataloader (here, augmentation is False)
    valid_bounds = (int(n_samples * tr_ratio), None)  # None means very last one
    valid_dataset = SegmentationDataset(dataset_path, valid_bounds, n_classes, False, remove_ground, speedup_factor)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=batch_size_valid, shuffle=True)

    # Return the dataloaders to the computer
    return train_dataloader, valid_dataloader
