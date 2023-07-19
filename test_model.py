import torch
import torchvision.transforms as IT
import os
import numpy as np
from PIL import Image
from src.load_inference import load_inference_model
import re

from src.dataset_fn_rl import DATASET_MEAN, DATASET_STD

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

def transform_img(samples):
        ''' Transform the samples.
        
        Parameters
        ----------
        samples : np.array
            Samples to transform.

        Returns
        -------
        samples : torch.Tensor
            Transformed samples.
        '''

        samples = samples.transpose((2, 0, 1, 3))
        sample_list = [samples[..., t] for t in range(samples.shape[-1])]

        # crop_params = IT.RandomCrop.get_params(torch.tensor(sample_list[0]),
        #                                        output_size=(240, 240))
        resize = IT.Resize(size=(128, 128),
                           interpolation=IT.InterpolationMode.NEAREST)
        normalize = IT.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
        jit = IT.ColorJitter(brightness=0.4,
                             contrast=0.4,
                             saturation=0.4,
                             hue=0.4)
        jit_params = IT.ColorJitter.get_params(jit.brightness,
                                               jit.contrast,
                                               jit.saturation,
                                               jit.hue)
        samples_and_labels = []
        for sample in sample_list:
            sample = torch.tensor(sample).to(torch.float32).div(255)
            sample = normalize(resize(sample))
            samples_and_labels.append((sample, None))

        samples = torch.stack([item[0] for item in samples_and_labels], dim=-1)
        return samples

@torch.inference_mode()
def test_model(model, device, input_sequence_dir, out_dir="test_results"):
    imgs = os.listdir(input_sequence_dir)
    imgs.sort(key=natural_keys)

    input_sequence = np.stack(tuple([np.array(Image.open(os.path.join(input_sequence_dir, i))) for i in imgs]), axis=-1)
    dim = input_sequence.shape[0:2]
    input_sequence = transform_img(input_sequence)
    input_sequence = input_sequence[None, ...]
    input_sequence = input_sequence.to(device='cuda')

    os.makedirs(out_dir, exist_ok=True)
    transform_stack = IT.Resize(size=dim, interpolation=IT.InterpolationMode.NEAREST)
    transform_pil = IT.ToPILImage(mode="RGB")

    E_seq, P_seq, S_seq = [], [], []
    num_frames = input_sequence.shape[-1]
    for t in range(num_frames):
        input_image = input_sequence[..., t]
        E, P, S = model(input_image, t)

        pimg = transform_pil(transform_stack(P[0, ...]))
        pimg.save(os.path.join(out_dir, f"pred_{t}.png"))

        tfimg = torch.zeros((3,128,128), device=device)
        tfimg[1:3,:,:] = S[0, ...]
        mimg = transform_pil(transform_stack(tfimg))
        mimg.save(os.path.join(out_dir, f"mask_{t}.png"))

        E_seq.append(E); P_seq.append(P); S_seq.append(S)

    return E_seq, P_seq, S_seq

if __name__ == "__main__":
    model, device = load_inference_model("img_seg.ckpt")
    test_model(model, device=device, input_sequence_dir="test_in", out_dir="test_out")