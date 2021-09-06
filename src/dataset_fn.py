import torch.utils.data as data
import torchvideo.transforms as VT
import torchvision.transforms as IT
import os
import random
from PIL import Image
VGG_MEAN = [0.0, 0.0, 0.0]
VGG_STD = [1.0, 1.0, 1.0]


class Mots_Dataset(data.Dataset):
  def __init__(self, root_dir, n_frames):
    self.sample_root = os.path.join(root_dir, 'training', 'image_02')
    self.label_root = os.path.join(root_dir, 'instances', 'instances')
    self.n_frames = n_frames  # idea: variable n_frames for each batch?
    self.sample_transform = IT.Compose([
      IT.ToTensor(),
      IT.Normalize(mean=VGG_MEAN, std=VGG_STD)])
    self.label_transform = IT.ToTensor()

  def __len__(self):
    pass  # to do

  def __getitem__(self, index):
    sample_dir = os.path.join(self.sample_root, f'{index:04}')
    label_dir = os.path.join(self.label_root, f'{index:04}')
    sample_paths = [os.path.join(sample_dir, p) for p in os.listdir(sample_dir)]
    label_paths = [os.path.join(label_dir, p) for p in os.listdir(label_dir)]

    first_frame = random(0, len(sample_paths) - self.n_frames)
    sample_paths = sample_paths[first_frame:first_frame + self.n_frames]
    label_paths = label_paths[first_frame:first_frame + self.n_frames]
    
    samples = self.sample_transform([Image.open(os.path.join(p)) for p in sample_paths])       # check if transform(list of PIL) is possible? and what is the output?
    labels = self.label_transform([Image.open(os.path.join(p)) for p in label_paths]) // 1000  # espescially output dims? might have to use permute
    labels[labels == 10] = 0

    return samples.cuda(), labels.cuda()


def get_datasets_seg(
  root_dir, train_valid_ratio,
  batch_size_train, batch_size_valid, n_frames):
  
  # Data train valid
  train_subfolders = []

  # Training dataloader
  train_dataset = Mots_Dataset(root_dir, train_subfolders, n_frames)
  train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size_train, shuffle=True)

  # Validation dataloader
  valid_dataset = Mots_Dataset(root_dir, valid_subfolders, n_frames)
  valid_dataloader = data.DataLoader(
    valid_dataset, batch_size=batch_size_valid, shuffle=True)  # todo: define what subfolders are linked to valid and what are linked to train

  # Return the dataloaders to the computer
  return train_dataloader, valid_dataloader


def main():
  root_dir = os.path.join('D:\\', 'DL', 'datasets', 'kitti', 'mots')
  batch_size_train = 1
  batch_size_valid = 1
  train_valid_ratio = 0.99
  n_frames = 20
  train_dl, valid_dl = get_datasets_seg(
    root_dir, train_valid_ratio,
    batch_size_train, batch_size_valid, n_frames)
  import matplotlib.pyplot as plt
  for sample, label in valid_dl:
    label = label.cpu().numpy()
    plt.imshow(label[0, 0])
    plt.show()

if __name__ == '__main__':
  main()
