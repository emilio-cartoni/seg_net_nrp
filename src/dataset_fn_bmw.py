import torch.utils.data as data
import numpy as np
import torchvision.transforms as IT
import torchvision.transforms.functional as TF
import os
import random
import torch
from PIL import Image
# DATASET_MEAN = [0.38, 0.38, 0.38]
# DATASET_STD = [0.28, 0.28, 0.28]
DATASET_MEAN = [0.00, 0.00, 0.00]  # use this to have data between 0.0 and 1.0
DATASET_STD = [1.00, 1.00, 1.00]  # use this to have data between 0.0 and 1.0


class BMW_Dataset(data.Dataset):
    def __init__(self, sample_dir, label_dir, labels, categories, subfolders, n_classes, n_frames, remove_ground):
        super(BMW_Dataset, self).__init__()
        self.sample_dir = sample_dir
        self.label_dir = label_dir
        self.labels = ['_skeleton__' + s for s in labels]
        self.categories = {k: ['_skeleton__' + v for v in vv] for k, vv in categories.items()}
        self.subfolders = subfolders
        self.n_classes = n_classes
        self.n_frames = n_frames  # idea: variable n_frames for each batch?
        self.remove_ground = remove_ground

    def transform(self, image_list, label_dict_list):
        transformed_images_and_labels = []
        # resize = IT.Resize(size=(120, 120), interpolation=IT.InterpolationMode.NEAREST)
        # i, j, h, w = IT.RandomCrop.get_params(image_list[0], output_size=(240, 240))
        resize = IT.Resize(size=(128, 128), interpolation=IT.InterpolationMode.NEAREST)
        normalize = IT.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
        for image, label_dict in zip(image_list, label_dict_list):
            # image = TF.crop(image, i, j, h, w)
            image = resize(image)
            image_tensor = normalize(TF.to_tensor(np.array(image)))
            labels = []
            for category_content in self.categories.values():
                to_append = 0.0  # ok with addition with shaped tensor
                for label in self.labels:
                    if label in category_content:
                        label_pil = label_dict[label]
                        # label_pil = TF.crop(label_pil, i, j, h, w)
                        label_pil = resize(label_pil)
                        to_append += TF.to_tensor(np.array(label_pil)[:, :, -1])
                labels.append(torch.minimum(torch.ones_like(to_append), to_append))
            label_tensor = torch.stack(labels, dim=0)
            transformed_images_and_labels.append((image_tensor, label_tensor))
        return transformed_images_and_labels

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, index):
        sample_subpath = f'cam_img_{self.subfolders[index]:05}'
        label_subpath = f'seg_img_{self.subfolders[index]:05}'
        image_path_list = [path for path in os.listdir(self.sample_dir) if sample_subpath in path]
        label_path_dict = {label: [path for path in os.listdir(self.label_dir) if label_subpath in path and label in path] for label in self.labels}
        n_frames = min(self.n_frames, len(image_path_list) - 1)  # - 1 or not (?)
        first_frame = random.choice(np.arange(len(image_path_list) - n_frames))
        image_path_list = image_path_list[first_frame:first_frame + n_frames]
        label_path_dict = {label: label_path_dict[label][first_frame:first_frame + n_frames] for label in self.labels}

        image_list, label_dict_list = [], []
        for i, image_path in enumerate(image_path_list):
            image_full_path = os.path.join(self.sample_dir, image_path)
            image = Image.open(image_full_path)  # for now, just an RGB image
            label_dict = {}
            for label in self.labels:
                label_full_path = os.path.join(self.label_dir, label_path_dict[label][i])
                label_dict[label] = Image.open(label_full_path)
            image_list.append(image)
            label_dict_list.append(label_dict)
        samples_and_labels = self.transform(image_list, label_dict_list)

        samples = torch.stack([item[0] for item in samples_and_labels], dim=-1)
        labels = torch.stack([item[1] for item in samples_and_labels], dim=-1)
        labels[labels == 10] = self.n_classes - int(not self.remove_ground)
        labels = torch.squeeze(labels, dim=1)
        # if not self.remove_ground:
        #     create ground tensor by taking the sum over all
        #     and then 1 minus that and them max of that vs zeros
        #     then concatenate it at index 0
        return samples, labels


def get_bmw_dataloaders(
    root_dir, train_valid_ratio, batch_size_train, batch_size_valid, n_frames, augmentation, remove_ground):
  
    # Dataset info
    sample_dir = os.path.join(root_dir, 'camera_images')
    label_dir = os.path.join(root_dir, 'segment_images')
    n_indices = len(set([path[8:13] for path in os.listdir(sample_dir)]))
    train_indices = list(range(0, int(n_indices * train_valid_ratio)))
    valid_indices = list(range(int(n_indices * train_valid_ratio), n_indices))
    label_names = ['demo_screwdriver__link', 'skeleton',
                   'forearm', 'upper_arm', 'hand',
                   'distal', 'middle', 'proximal',
                   'thumb_1', 'thumb_2', 'thumb_3']
    label_categories = {'arm': ['forearm', 'upper_arm'],
                        'hand': ['hand', 'distal', 'middle', 'proximal', 'thumb_1', 'thumb_2', 'thumb_3'],
                        'tool': ['demo_screwdriver__link'],
                        'torso': ['skeleton']}
    n_classes = len(label_categories.keys()) + int(not remove_ground)

    # Training dataloader
    train_dataset = BMW_Dataset(sample_dir, label_dir, label_names, label_categories, train_indices, n_classes, n_frames, remove_ground)
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True)

    # Validation dataloader
    valid_dataset = BMW_Dataset(sample_dir, label_dir, label_names, label_categories, valid_indices, n_classes, n_frames, remove_ground)
    valid_dataloader = data.DataLoader(
        valid_dataset, batch_size=batch_size_valid, shuffle=True, pin_memory=True)

    # Return the dataloaders to the computer, also returns the number of classes implicitely
    return train_dataloader, valid_dataloader, n_classes
