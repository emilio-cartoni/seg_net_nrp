import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# directory = r'C:\Users\loennqvi\Github\seg_net_vgg\data\SQM\testing\0003'
directory = r'D:\DL\datasets\kitti\mots\sqm\testing\0000'
image_size = (160, 512)
n_channels = 3
line_luminance = 255
line_height = 30
line_width = 3
vertical_gap = 3
stream_speed = 5
frame_offsets = {0: 9, 5: -9}
n_blank_frames = 20
n_sqm_frames = 40
n_frames = 2 * (n_blank_frames) + n_sqm_frames
make_white_background = False

def sqm_patch(
    line_height, line_width, line_luminance, vertical_gap, offset):

    patch_size = (2 * line_height + vertical_gap, line_width + abs(offset))
    patch = np.zeros(patch_size + (n_channels,))
    patch[:line_height, :line_width, :] = line_luminance
    patch[-line_height:, -line_width:, :] = line_luminance
    return patch if offset >= 0 else np.fliplr(patch)


for frame in range(n_frames):

    image = np.zeros(image_size + (n_channels,))
    #image = np.random.uniform(low=0.0, high=255., size=(image_size + (n_channels,)))
    if n_blank_frames <= frame < n_frames - n_blank_frames:
        sqm_frame = (frame - n_blank_frames) // 1
        if 1:  # (frame % 2) == ((n_blank_frames % 2) != 0):
        
            try:
                offset = frame_offsets[sqm_frame]
            except KeyError:
                offset = 0
            
            left_patch = sqm_patch(line_height, line_width, line_luminance, vertical_gap, offset)
            right_patch = sqm_patch(line_height, line_width, line_luminance, vertical_gap, 0)

            first_row = image.shape[0] // 2 - left_patch.shape[0] // 2
            first_col = image.shape[1] // 2 - left_patch.shape[1] // 2 + stream_speed * sqm_frame
            last_row = first_row + left_patch.shape[0]
            last_col = first_col + left_patch.shape[1]
            image[first_row:last_row, first_col:last_col, :] = left_patch

            if sqm_frame > 0:
                first_row = image.shape[0] // 2 - right_patch.shape[0] // 2
                first_col = image.shape[1] // 2 - right_patch.shape[1] // 2 - stream_speed * sqm_frame
                last_row = first_row + right_patch.shape[0]
                last_col = first_col + right_patch.shape[1]
                image[first_row:last_row, first_col:last_col, :] = right_patch

    if make_white_background:
        image[image == 255.] = 1.
        image[image == 0.] = 255.
        image[image == 1.] = 0.
    image = image.astype(np.uint8)
    # image = image - 1
    # image[image == 0] = 1
    # image = image * 255
    # plt.imshow(image)
    # plt.show()
    file_path = f'{directory}\\{frame:06}.png'
    image = Image.fromarray(image, mode='RGB')
    image.save(file_path)
    