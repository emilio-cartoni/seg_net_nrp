import os
import numpy as np
from PIL import Image
from itertools import repeat
from multiprocessing import Pool, cpu_count
import random

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

def save_mask(img, img_path, lbl_dict, mask_id):
    ''' Save the images and labels to the segment directory.

    '''
    np_img = np.array(img)
    img_shape = np_img.shape
    mask_shape = img_shape[:-1] + (len(lbl_dict.keys()),)
    segment_mask_img = np.zeros(mask_shape, dtype=np.uint8)

    mask = np.where(np.add(np.add(np_img[:, :, 0], np_img[:, :, 1]), np_img[:, :, 2]) > 0, 1, 0).astype(np.uint8)
    segment_mask_img[:,:, mask_id] = np.logical_or(segment_mask_img[:,:, mask_id], mask)

    Image.fromarray(mask*255, mode='L').save(f"{img_path}.png")

    np.save(img_path, segment_mask_img)

    return mask

def parse_dataset(data_dir, num_sequences=2000):
    ''' Parse the dataset into a dictionary of images and their labels.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset directory.

    '''
    img_dir = os.path.join(data_dir, 'img')
    mask_dir = os.path.join(data_dir, 'mask')
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    if len(os.listdir(img_dir)) >= num_sequences:
        return

    lbl_dict = { 
        'chips_can': 'chips_can', 
        'mustard': 'mustard', 
        # 'plate': 'plate', 
        #'screwdriver': 'screwdriver' 
    }
    
    num_processors_used = 8  # min(cpu_count() - 2, 8)
    print(f'Using {num_processors_used} processors')
    # for i in range(0, num_sequences): 
    #     parse_one_sequence(data_dir, img_dir, mask_dir, lbl_dict, i, num_sequences)
    with Pool(num_processors_used) as pool:
        pool.starmap(parse_one_sequence, zip(repeat(data_dir),
                                             repeat(img_dir),
                                             repeat(mask_dir),
                                             repeat(lbl_dict),
                                             [ i for i in range(0, num_sequences) ],
                                             repeat(num_sequences)))

def parse_one_sequence(data_dir, img_dir,
                       mask_dir, lbl_dict,
                       cur_sequence,
                       num_sequences):
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
    num_sequences_done = len(os.listdir(img_dir))
    num_sequences_todo = num_sequences
    print(f'\rDoing sequence {num_sequences_done}/{num_sequences_todo}', end='')

    img_total_sum = 540
    img_per_orig_sequence = 90
    img_per_new_sequence = 20

    orig_img_start = random.randint(0, img_per_orig_sequence - img_per_new_sequence-1)
    orig_seq = random.randint(0, int(round(img_total_sum / img_per_orig_sequence))-1)
    obj_id = random.randint(0, len(lbl_dict.values())-1)

    obj_dir = list(lbl_dict.values())[obj_id]
    src_img_dir = os.path.join(data_dir, obj_dir, 'image')
    src_mask_dir = os.path.join(data_dir, obj_dir, 'segmented')

    img_subdir = os.path.join(img_dir, str(cur_sequence))
    mask_subdir = os.path.join(mask_dir, str(cur_sequence))

    os.makedirs(img_subdir, exist_ok=True)
    os.makedirs(mask_subdir, exist_ok=True)

    for i in range(orig_img_start, orig_img_start + img_per_new_sequence):
        src_img_name = f"{orig_seq*img_per_orig_sequence + i}-pose{orig_seq}.png"
        new_img_name = f"image_{i - orig_img_start}"

        with Image.open(os.path.join(src_img_dir, src_img_name)) as new_img:
            new_img.save(os.path.join(img_subdir, f"{new_img_name}.png"))

        with Image.open(os.path.join(src_mask_dir, src_img_name)) as new_mask:
            save_mask(new_mask, os.path.join(mask_subdir, f"{new_img_name}.npy"), lbl_dict, obj_id)
