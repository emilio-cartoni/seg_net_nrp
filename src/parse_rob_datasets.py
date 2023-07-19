import os
import numpy as np
from PIL import Image
from itertools import repeat
from multiprocessing import Pool, cpu_count

def save_imgs(img_dict, lbl_dict, segment_dir_out, packbits=True):
    ''' Save the images and labels to the segment directory.

    Parameters
    ----------
    img_dict : dict
        Dictionary of camera images.
    lbl_dict : dict
        Dictionary of segmentation images.
    segment_dir_out : str
        Path to the parsed segmented masks directory.
    packbits : bool
        Whether to use packbits compression or not.

    '''
    first_img_path = list(img_dict.keys())[0]
    img_shape = img_dict[first_img_path].shape
    mask_shape = img_shape[:-1] + (len(lbl_dict.keys()),)
    segment_mask_img = np.zeros(mask_shape, dtype=np.uint8)
    segment_mask_path = first_img_path[:22] + '.npy'

    segment_subdir_out = segment_mask_path.split('_')[-2]
    full_segment_mask_path = os.path.join(segment_dir_out,
                                          segment_subdir_out,
                                          segment_mask_path)

    for img_path, img in img_dict.items():
        this_label = img_path.split('__')[-1].split('.png')[0]
        #print(img_path)
        for cat_idx, cat in enumerate(lbl_dict.keys()):
            
            if this_label in lbl_dict[cat]:
                this_mask = np.where(img[:, :, 0] > 0, 1, 0).astype(np.uint8)
                segment_mask_img[:, :, cat_idx] += this_mask

    if packbits:
        segment_mask_img = np.packbits(segment_mask_img, axis=1)
    np.save(full_segment_mask_path, segment_mask_img)

def parse_dataset(data_dir, dataset_type, packbits):
    ''' Parse the dataset into a dictionary of images and their labels.

    Parameters
    ----------
    dataset_type : str
        Type of dataset to parse.
    dataset_dir : str
        Path to the dataset directory.

    '''
    segment_subdir = 'sb' if packbits else 'sm'
    segment_dir_in = os.path.join(data_dir, dataset_type, 'si')
    segment_dir_out = os.path.join(data_dir, dataset_type, segment_subdir)
    os.makedirs(segment_dir_out, exist_ok=True)

    if dataset_type == 'rob':
        lbl_dict = {
            'iiwa': [f'iiwa_link_{i}' for i in range(8)],
            'torso': ['skeleton'],
            'upper_arm': ['upper_arm'],
            'forearm': ['forearm'],
            'hand': [
                'hand', 'distal', 'middle', 'proximal',
                'thumb_1', 'thumb_2', 'thumb_3']}

    elif dataset_type == 'no_rob':
        lbl_dict = {
            'torso': ['skeleton'],
            'upper_arm': ['upper_arm'],
            'forearm': ['forearm'],
            'hand': [
                'hand', 'distal', 'middle', 'proximal',
                'thumb_1', 'thumb_2', 'thumb_3']}

    else:
        raise SystemExit('\nError: invalid dataset type')
    
    num_processors_used = 8  # min(cpu_count() - 2, 8)
    print(f'Using {num_processors_used} processors')
    segment_subdirs_in = sorted(os.listdir(segment_dir_in))
    with Pool(num_processors_used) as pool:
        pool.starmap(parse_one_sequence, zip(segment_subdirs_in,
                                             repeat(segment_dir_in),
                                             repeat(segment_dir_out),
                                             repeat(lbl_dict),
                                             repeat(packbits)))

def parse_one_sequence(segment_subdir_in, segment_dir_in,
                       segment_dir_out, lbl_dict, packbits):
    ''' Parse one sequence of images and their labels.

    Parameters
    ----------
    segment_subdir_in : str
        Name of the subdirectory being processed
    segment_dir_in : str
        Path to the directory containing the segmentation images.
    segment_dir_out : str
        Path to the directory that will contain the segmentation masks.
    lbl_dict : dict
        Dictionary of labels used to create the masks.

    '''
    num_sequences_done = len(os.listdir(segment_dir_out))
    num_sequences_todo = len(os.listdir(segment_dir_in))
    print(f'\rDoing sequence {num_sequences_done}/{num_sequences_todo}', end='')

    full_segment_subdir_in = os.path.join(segment_dir_in, segment_subdir_in)
    full_segment_subdir_out = os.path.join(segment_dir_out, segment_subdir_in)
    try:
        os.makedirs(full_segment_subdir_out, exist_ok=False)
    except FileExistsError:
        num_images_per_subdir = len(os.listdir(full_segment_subdir_in))
        num_labels_per_timestep = sum([len(v) for v in lbl_dict.values()])
        num_time_steps = num_images_per_subdir // num_labels_per_timestep - 1
        if len(os.listdir(full_segment_subdir_out)) < num_time_steps:
            pass
        else:
            return
    
    mask_dict = {}
    check_path = ''
    for img_path in sorted(os.listdir(full_segment_subdir_in)):
        new_check_path = img_path[:22]

        if new_check_path != check_path and check_path != '':
            save_imgs(mask_dict, lbl_dict, segment_dir_out, packbits)
            mask_dict = {}
            check_path = ''
        
        check_path = new_check_path
        full_img_path = os.path.join(full_segment_subdir_in, img_path)

        with Image.open(full_img_path) as new_img:
            mask_dict[img_path] = np.array(new_img)
