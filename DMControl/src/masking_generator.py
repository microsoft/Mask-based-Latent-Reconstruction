# --------------------------------------------------------
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import random
import math
import numpy as np
from numpy.core.shape_base import block
import torch
import torch.nn as nn


class MaskingGenerator:
    ''' Borrowed from https://github.com/pengzhiliang/MAE-pytorch '''
    def __init__(
            self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
            min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask

class RandomMaskingGenerator:
    ''' Borrowed from https://github.com/pengzhiliang/MAE-pytorch '''
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask.astype(np.bool) # [196]

class RandomMaskingMapGenerator:
    def __init__(self, input_size, mask_ratio, image_size):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

        self.image_size = image_size
        self.upsampler = nn.Upsample((image_size, image_size))

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        mask = torch.from_numpy(mask).reshape(self.height, self.width)
        mask = self.upsampler(mask[None, None].float())
        return mask # [196]

class DiverseRandomMaskingMapGenerator:
    def __init__(self, input_size, mask_ratio, image_size, time_span):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

        self.image_size = image_size
        self.upsampler = nn.Upsample((image_size, image_size))

        self.time_span = time_span

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        mask = np.vstack([mask] * self.time_span)

        for i in range(self.time_span):
            mask[i] = mask[i, torch.randperm(mask.shape[-1])]

        mask = torch.from_numpy(mask)[:, :, None].reshape(self.time_span, self.height, self.width)
        mask = self.upsampler(mask[None].float())
        return mask

class RandomMaskingListGenerator:
    def __init__(self, list_len, mask_ratio):
        self.num_patches = list_len
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask.astype(np.bool)
        # return torch.from_numpy(mask).float() # [196]


class RestMaskingListGenerator:
    def __init__(self, list_len, mask_ratio):
        self.num_patches = list_len
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        # np.random.shuffle(mask)
        return mask.astype(np.bool)
        # return torch.from_numpy(mask).float() # [196]

class RandomRestMaskingListGenerator:
    '''
    t > offset, set to True; t < offset, set to False 
    '''
    def __init__(self, list_len):
        self.num_patches = list_len

    def __repr__(self):
        repr_str = "Maks: total patches {}".format(
            self.num_patches
        )
        return repr_str

    def __call__(self):
        offset = np.random.randint(1, self.num_patches)
        mask = np.hstack([
            np.zeros(offset),
            np.ones(self.num_patches - offset),
        ])
        # np.random.shuffle(mask)
        return mask.astype(np.bool)
        # return torch.from_numpy(mask).float() # [196]

class RandomBlockMaskingListGenerator:
    def __init__(self, list_len, mask_ratio, block_size):
        assert list_len % block_size == 0
        self.list_len = list_len
        self.block_size = block_size
        self.num_blocks = list_len // block_size

        self.num_mask = int(mask_ratio * self.num_blocks)

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_blocks - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        mask = np.repeat(mask, self.block_size)
        return mask.astype(np.bool)

class RandomMaskGenerator:
    def __init__(self, list_len, block_size, mask_ratio):
        assert mask_ratio <= 1.0
        self.block_size = block_size
        self.num_blocks = list_len // block_size
        self.num_masked_blocks = int(mask_ratio * self.num_blocks)
        self.num_rest = list_len % block_size
    
    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_blocks - self.num_masked_blocks),
            np.ones(self.num_masked_blocks),
        ])
        np.random.shuffle(mask)
        mask = np.repeat(mask, self.block_size)
        mask = np.concatenate([mask, np.zeros(self.num_rest)], 0)
        mask = np.roll(mask, np.random.randint(0, self.block_size))
        # mask[0] = 0 # set the first elsement always to unmasked
        return mask.astype(np.bool)

class CubeMaskGenerator:
    def __init__(self, input_size, image_size, clip_size, block_size, mask_ratio):
        assert mask_ratio <= 1.0

        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)
        self.image_size = image_size
        self.upsampler = nn.Upsample((image_size, image_size))

        self.block_size = block_size
        self.num_blocks = clip_size // block_size

    
    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        for i in range(self.num_blocks):
            np.random.shuffle(mask)
            cur_mask = torch.from_numpy(mask).reshape(self.height, self.width)
            cur_mask = self.upsampler(cur_mask[None, None].float()) # (1, 1, h, w)
            cur_mask = cur_mask.expand(self.block_size, *cur_mask.size()[1:])
            cube_mask = torch.cat([cube_mask, cur_mask]) if i > 0 else cur_mask
        return cube_mask

if __name__ == '__main__':
    masker = DiverseRandomMaskingMapGenerator(4, 0.5, 8, 2)
    print(masker())