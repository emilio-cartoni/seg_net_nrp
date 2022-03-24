import torch.utils.data as data
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import os
import random
import torch
import matplotlib.pyplot as plt
from pycocotools.ytvos import YTVOS
from PIL import Image
DATASET_MEAN = [0.00, 0.00, 0.00]  # use this to have data between 0.0 and 1.0
DATASET_STD = [1.00, 1.00, 1.00]  # use this to have data between 0.0 and 1.0


class Ovis_Dataset(data.Dataset):
    def __init__(self, data_root_dir, vos, vid_ids, n_classes, n_frames, remove_ground):
        # Initialize dataset and annotations (compressed segmentation masks)
        self.data_samples_dir = os.path.join(data_root_dir, 'train')  # even for valid
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.remove_ground = remove_ground
        self.vos = vos
        self.vid_ids = vid_ids

    def transform(self, list_of_samples_and_labels):
        """
        Parameters
        ----------
        list_of_samples_and_labels : list
        Must be a list of tuples, where each tuple inside the list contains both
        a training image and its corresponding label.

        Returns
        -------
        transformed_samples_and_labels : list
        A list of tuples, where each tuple is a set of (image, label) for a specific timepoint.
        """
        # Parameters of the transformation
        transformed_samples_and_labels = []
        crop_shape = (768, 768)
        resize_shape = (192, 192)

        # Check that the image is not too small and resize in that case
        sample_w, sample_h = list_of_samples_and_labels[0][0].size  # careful order (w, h)
        ratios = sample_h / crop_shape[0], sample_w / crop_shape[1]
        if any([r < 1 for r in ratios]):
            shape_to_match = crop_shape[np.argmin(ratios)]
            correct = T.Resize(size=shape_to_match, interpolation=T.InterpolationMode.NEAREST)
            new_list_of_samples_and_labels = []
            for sample, label in list_of_samples_and_labels:
                sample, label = correct(sample), correct(label)
                new_list_of_samples_and_labels.append((sample, label))
            list_of_samples_and_labels = new_list_of_samples_and_labels

        # Create the transformation objects
        resize = T.Resize(size=resize_shape, interpolation=T.InterpolationMode.NEAREST)
        normalize = T.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
        i, j, h, w = T.RandomCrop.get_params(list_of_samples_and_labels[0][0],
                                              output_size=crop_shape)

        # Transform the samples and labels and send back to the computer
        for sample, label in list_of_samples_and_labels:
            sample, label = TF.crop(sample, i, j, h, w), TF.crop(label, i, j, h, w)
            sample, label = resize(sample), resize(label)
            sample, label = TF.to_tensor(np.array(sample)), TF.to_tensor(np.array(label))
            sample = normalize(sample)
            transformed_samples_and_labels.append((sample, label))
        return transformed_samples_and_labels

    def __len__(self):
        return len(self.vid_ids)

    def __getitem__(self, index):
        # Load video and corresponding annotations
        frames = self.vos.loadVids(self.vid_ids[index])[0]
        annIds = self.vos.getAnnIds(vidIds=frames['id'], iscrowd=None)
        anns = self.vos.loadAnns(annIds)

        # Load and display the video and the corresponding mask video
        samples_and_labels = []
        
        # for frame_id, frame_path in enumerate(frames['file_names']):
        for frame_id in range(self.n_frames):
            frame_id_used = frame_id % len(frames['file_names'])
            frame_path = frames['file_names'][frame_id_used]

            sample = Image.open(os.path.join(self.data_samples_dir, frame_path))
            visible_anns = [ann for ann in anns if ann['segmentations'][frame_id_used] is not None]
            masks = [self.vos.annToMask(ann, frame_id_used) for ann in visible_anns]
            cats = [ann['category_id'] for ann in visible_anns]
            segmented_masks = [cat * mask for cat, mask in zip(cats, masks)]
            try:
                label = Image.fromarray(np.array(segmented_masks).sum(axis=0))
            except IndexError:
                for s in segmented_masks:
                    print(s.max(), s.min(), s.shape)
            samples_and_labels.append((sample, label))

        # Transform samples and labels and prepare them for the network
        samples_and_labels = self.transform(samples_and_labels)
        samples = torch.stack([item[0] for item in samples_and_labels], dim=-1)
        labels = torch.stack([item[1] for item in samples_and_labels], dim=-1)
        labels = torch.nn.functional.one_hot(labels.to(torch.int64), 
                                             num_classes=self.n_classes + int(self.remove_ground))
        labels = torch.movedim(labels, -1, 1)
        labels = torch.squeeze(labels, dim=0)
        labels = labels.type(torch.FloatTensor)
        if self.remove_ground:
            labels = labels[1:]
        return samples, labels


def get_ovis_dataloaders(data_root_dir, tr_ratio, batch_size_train, batch_size_valid,
                         n_frames, augmentation, remove_ground):

    # Split the dataset into a validation dataset and a training dataset
    annotation_subdirs = 'annotations', f'annotations_train.json'
    annotation_file_path = os.path.join(data_root_dir, *annotation_subdirs)
    vos = YTVOS(annotation_file_path)
    n_classes = len(vos.loadCats(vos.getCatIds())) + int(not remove_ground)
    all_ids = vos.getVidIds()
    tr_limit = int(tr_ratio * len(all_ids))
    train_ids = all_ids[:tr_limit]
    valid_ids = all_ids[tr_limit:]

    # Training dataloader
    train_dataset = Ovis_Dataset(data_root_dir, vos, train_ids,
                                 n_classes, n_frames, remove_ground)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size_train,
                                       shuffle=True, pin_memory=True)

    # Validation dataloader
    valid_dataset = Ovis_Dataset(data_root_dir, vos, valid_ids,
                                 n_classes, n_frames, remove_ground)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=batch_size_valid,
                                       shuffle=True, pin_memory=True)

    # Return the dataloaders and the number of classes to the computer
    return train_dataloader, valid_dataloader, n_classes
