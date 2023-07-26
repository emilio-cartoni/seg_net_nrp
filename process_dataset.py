import os
import numpy as np
import hickle as hkl
from PIL import Image
import glob

os.environ['CUDA_VISIBLE_DEVICES'] = ''

DATA_DIR = '/media/ibrahim/DataIbrahim/data/rl_dataset/'  # where raw data is stored (each sequence in a folder)
desired_im_sz = (120, 160)  # to what size should images be converted  - height then width

# Sequences used for validation and testing.
# Chosen randomly.
all_files = []
for file in glob.glob(DATA_DIR + 'img/*'):
    all_files.append(file.split('/')[-1])
np.random.shuffle(all_files)

val_recordings = all_files[0:100]
test_recordings = all_files[100:500]
train_recordings = all_files[500:]


# Create image datasets.
# Process images and saves them in train, val, test splits.
def process_data():
    splits = {s: [] for s in ['train', 'test', 'val']}
    splits['val'] = val_recordings
    splits['test'] = test_recordings
    splits['train'] = train_recordings

    for split in splits:
        im_list = []
        source_list = []  # corresponds to recording that image came from
        for folder in splits[split]:
            im_dir = os.path.join(DATA_DIR, 'img/', str(folder))
            files = list(os.walk(im_dir))[-1][-1]
            im_list += [im_dir + '/' + f for f in sorted(files)]
            source_list += [int(folder)] * len(files)

        print('Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
        X = np.zeros((2,) + (len(im_list),) + desired_im_sz + (3,), np.uint8)
        for i, im_file in enumerate(im_list):

            # Save image in first position
            im = Image.open(im_file)
            X[0, i] = process_im(im, desired_im_sz)

            # Save mask in second position
            mask_file = im_file.replace('img', 'mask').replace('.png', '.npy')
            mask = np.load(mask_file)
            mask = np.concatenate([mask, np.zeros((480, 640, 1))], axis=2) * 255
            mask = mask.astype(np.uint8)
            mask = Image.fromarray(mask)
            mask = process_im(mask, desired_im_sz)
            X[1, i] = mask

        # Create hickle files from data
        hkl.dump(X, os.path.join(DATA_DIR, 'X_' + split + '.hkl'))
        hkl.dump(source_list, os.path.join(DATA_DIR, 'sources_' + split + '.hkl'))


# resize
def process_im(im, desired_sz):
    im = im.resize((desired_sz[1], desired_sz[0]))
    im = np.array(im)
    return im


if __name__ == '__main__':
    process_data()
