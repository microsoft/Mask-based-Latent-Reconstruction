# --------------------------------------------------------
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import os
import random
import time
from collections import deque

import gym
import numpy as np
from numpy.core.fromnumeric import resize
import torch
import torch.nn as nn
from skimage.util.shape import view_as_windows
from torch.utils.data import DataLoader, Dataset
import math


class InverseSquareRootSchedule(object):

    def __init__(self, warmup_step=4e4):
        if warmup_step is None:
            self.warmup_step = warmup_step
        else:
            warmup_step = int(warmup_step)
            assert warmup_step > 0 and isinstance(warmup_step, int)
            self.warmup_step = warmup_step
            init = 5e-4
            end = 1
            self.init_lr = init
            self.lr_step = (end - init) / warmup_step
            self.decay = warmup_step ** 0.5

    def step(self, step):
        if self.warmup_step is None:
            return  1
        else:
            if step < self.warmup_step:
                return self.init_lr + self.lr_step * step
            else:
                return self.decay * (step ** -0.5)


class AnneallingSchedule(object):
    def __init__(self, warmup_step=4e4):
        if warmup_step is None:
            self.warmup_step = warmup_step
        else:
            warmup_step = int(warmup_step)
            assert warmup_step > 0 and isinstance(warmup_step, int)
            self.warmup_step = warmup_step
            self.decay = warmup_step ** 0.5

    def step(self, step):
        if self.warmup_step is None:
            return  1
        else:
            if step < self.warmup_step:
                return 1
            else:
                return self.decay * (step ** -0.5)

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=128):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length):
        return self.pe[:, :length]

def to_categorical(value, limit=300):
    value = value.float()  # Avoid any fp16 shenanigans
    value = value.clamp(-limit, limit)
    distribution = torch.zeros(value.shape[0], (limit * 2 + 1),
                               device=value.device)
    lower = value.floor().long() + limit
    upper = value.ceil().long() + limit
    upper_weight = value % 1
    lower_weight = 1 - upper_weight
    distribution.scatter_add_(-1, lower.unsqueeze(-1),
                              lower_weight.unsqueeze(-1))
    distribution.scatter_add_(-1, upper.unsqueeze(-1),
                              upper_weight.unsqueeze(-1))
    return distribution


def maybe_transform(image, transform, alt_transform, p=0.8):
    processed_images = transform(image)
    if p >= 1:
        return processed_images
    else:
        base_images = alt_transform(image)
        mask = torch.rand((processed_images.shape[0], 1, 1, 1),
                          device=processed_images.device)
        mask = (mask < p).float()
        processed_images = mask * processed_images + (1 - mask) * base_images
        return processed_images


def infer_leading_dims(tensor, dim):
    """Looks for up to two leading dimensions in ``tensor``, before
    the data dimensions, of which there are assumed to be ``dim`` number.
    For use at beginning of model's ``forward()`` method, which should 
    finish with ``restore_leading_dims()`` (see that function for help.)
    Returns:
    lead_dim: int --number of leading dims found.
    T: int --size of first leading dim, if two leading dims, o/w 1.
    B: int --size of first leading dim if one, second leading dim if two, o/w 1.
    shape: tensor shape after leading dims.
    """
    lead_dim = tensor.dim() - dim
    assert lead_dim in (0, 1, 2)
    if lead_dim == 2:
        T, B = tensor.shape[:2]
    else:
        T = 1
        B = 1 if lead_dim == 0 else tensor.shape[0]
    shape = tensor.shape[lead_dim:]
    return lead_dim, T, B, shape


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(self,
                 obs_shape,
                 action_shape,
                 capacity,
                 batch_size,
                 device,
                 image_size=84,
                 transform=None,
                 auxiliary_task_batch_size=64,
                 jumps=5):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        self.auxiliary_task_batch_size = auxiliary_task_batch_size
        self.jumps = jumps
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)  # for infinite bootstrap
        self.real_dones = np.empty((capacity, 1), dtype=np.float32) # for auxiliary task

        self.idx = 0
        self.last_save = 0
        self.full = False

        self.current_auxiliary_batch_size = batch_size

    def add(self, obs, action, reward, next_obs, done):

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)   # "not done" is always True
        np.copyto(self.real_dones[self.idx], isinstance(done, int))   # "not done" is always True
        # print(not done, isinstance(done, int))

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self):

        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=self.batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def sample(self):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=self.batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        curr_rewards = torch.as_tensor(self.curr_rewards[idxs],
                                       device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()

        return obses, actions, curr_rewards, rewards, next_obses, not_dones

    # v3
    def sample_idm(self):   # sample batch for auxiliary task
        jumps = 1
        idxs = np.random.randint(0,
                                 self.capacity - jumps -
                                 1 if self.full else self.idx - jumps - 1,
                                 size=self.auxiliary_task_batch_size*2)
                                #  size=self.auxiliary_task_batch_size)
        idxs = idxs.reshape(-1, 1)
        step = np.arange(jumps + 1).reshape(1, -1) # this is a range
        idxs = idxs + step

        real_dones = torch.as_tensor(self.real_dones[idxs], device=self.device)   # (B, jumps+1, 1)
        # we add this to avoid sampling the episode boundaries
        valid_idxs = torch.where((real_dones.mean(1)==0).squeeze(-1))[0].cpu().numpy()
        idxs = idxs[valid_idxs] # (B, jumps+1)
        idxs = idxs[:self.auxiliary_task_batch_size] if idxs.shape[0] >= self.auxiliary_task_batch_size else idxs
        self.current_auxiliary_batch_size = idxs.shape[0]

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()   # (B, jumps+1, 3*3=9, 100, 100)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)   # (B, jumps+1, 1)
    
        idm_samples = {
            'observation': obses.transpose(0, 1).unsqueeze(3),
            'action': actions.transpose(0, 1),
            'reward': rewards.transpose(0, 1),
        }
        
        return idm_samples

    # v2
    def sample_spr(self):   # sample batch for auxiliary task
        idxs = np.random.randint(0,
                                 self.capacity - self.jumps -
                                 1 if self.full else self.idx - self.jumps - 1,
                                 size=self.auxiliary_task_batch_size*2)
                                #  size=self.auxiliary_task_batch_size)
        idxs = idxs.reshape(-1, 1)
        step = np.arange(self.jumps + 1).reshape(1, -1) # this is a range
        idxs = idxs + step

        real_dones = torch.as_tensor(self.real_dones[idxs], device=self.device)   # (B, jumps+1, 1)
        # we add this to avoid sampling the episode boundaries
        valid_idxs = torch.where((real_dones.mean(1)==0).squeeze(-1))[0].cpu().numpy()
        idxs = idxs[valid_idxs] # (B, jumps+1)
        idxs = idxs[:self.auxiliary_task_batch_size] if idxs.shape[0] >= self.auxiliary_task_batch_size else idxs
        self.current_auxiliary_batch_size = idxs.shape[0]

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()   # (B, jumps+1, 3*3=9, 100, 100)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)   # (B, jumps+1, 1)
    
        spr_samples = {
            'observation': obses.transpose(0, 1).unsqueeze(3),
            'action': actions.transpose(0, 1),
            'reward': rewards.transpose(0, 1),
        }
        return (*self.sample_aug(original_augment=True), spr_samples)

    
    # # v1
    # def sample_spr(self):
    #     idxs = np.random.randint(0,
    #                              self.capacity - self.jumps -
    #                              1 if self.full else self.idx - self.jumps - 1,
    #                              size=self.auxiliary_task_batch_size)
    #     idxs = idxs.reshape(-1, 1)
    #     step = np.arange(self.jumps + 1).reshape(1, -1)
    #     idxs = idxs + step
    #     obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
    #     next_obses = torch.as_tensor(self.next_obses[idxs],
    #                                  device=self.device).float()
    #     actions = torch.as_tensor(self.actions[idxs], device=self.device)
    #     rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
    #     not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

    #     spr_samples = {
    #         'observation': obses.transpose(0, 1).unsqueeze(3),
    #         'action': actions.transpose(0, 1),
    #         'reward': rewards.transpose(0, 1),
    #     }
    #     # print(obses.transpose(0, 1).unsqueeze(3).size())
    #     return (*self.sample_aug(original_augment=True), spr_samples)
    

    def sample_aug(self, original_augment=False):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=self.batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        if not original_augment:
            obses = torch.as_tensor(obses, device=self.device).float()
            next_obses = torch.as_tensor(next_obses,
                                         device=self.device).float()
            if hasattr(self, 'SPR'):
                obses = self.SPR.transform(obses, augment=True)
                next_obses = self.SPR.transform(next_obses, augment=True)
            elif hasattr(self, 'CycDM'):
                obses = self.CycDM.transform(obses, augment=True)
                next_obses = self.CycDM.transform(next_obses, augment=True)
        else:
            # 1. Normal
            obses = random_crop(obses, self.image_size)
            next_obses = random_crop(next_obses, self.image_size)

            # # 2. Deterministic
            # obses = center_crop(obses, self.image_size)
            # next_obses = center_crop(next_obses, self.image_size)

            # # 3. Temporal Consistent
            # obses, next_obses = sync_crop(obses, self.image_size, next_obses)

            obses = torch.as_tensor(obses, device=self.device).float()
            next_obses = torch.as_tensor(next_obses,
                                         device=self.device).float()

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def sample_cpc(self):

        start = time.time()
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=self.batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()

        obses = random_crop(obses, self.image_size)
        next_obses = random_crop(next_obses, self.image_size)
        pos = random_crop(pos, self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        pos = torch.as_tensor(pos, device=self.device).float()
        cpc_kwargs = dict(obs_anchor=obses,
                          obs_pos=pos,
                          time_anchor=None,
                          time_pos=None)

        return obses, actions, rewards, next_obses, not_dones, cpc_kwargs

    def add_agent(self, agent):
        if hasattr(agent, 'CURL'):
            self.CURL = agent.CURL
        if hasattr(agent, 'SPR'):
            self.SPR = agent.SPR
        if hasattr(agent, 'CycDM'):
            self.CycDM = agent.CycDM

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(0,
                                self.capacity if self.full else self.idx,
                                size=1)
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k, ) + shp[1:]),
            dtype=env.observation_space.dtype)
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def sync_crop(imgs, output_size, next_imgs):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    assert imgs.shape == next_imgs.shape
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(imgs,
                              (1, output_size, output_size, 1))[..., 0, :, :,
                                                                0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]

    
    ''' process next observations '''
    next_imgs = np.transpose(next_imgs, (0, 2, 3, 1))
    next_windows = view_as_windows(next_imgs,
                              (1, output_size, output_size, 1))[..., 0, :, :,
                                                                0]
    # selects a random window for each batch element
    next_cropped_imgs = next_windows[np.arange(n), w1, h1]

    return cropped_imgs, next_cropped_imgs

def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(imgs,
                              (1, output_size, output_size, 1))[..., 0, :, :,
                                                                0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs

def center_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size

    # imgs = np.transpose(imgs, (0, 2, 3, 1))
    # w1 = crop_max // 2
    # h1 = crop_max // 2
    # # creates all sliding windows combinations of size (output_size)
    # windows = view_as_windows(imgs,
    #                           (1, output_size, output_size, 1))[..., 0, :, :,
    #                                                             0]
    # # selects a random window for each batch element
    # cropped_imgs = windows[np.arange(n), w1, h1]

    new_h, new_w = output_size, output_size
    top = (img_size - new_h) // 2
    left = (img_size - new_w) // 2
    cropped_imgs = imgs[:, :, top:top + new_h, left:left + new_w]

    # resized_imgs = np.resize(imgs, (n, 9, 84, 84))

    # import cv2
    # left_im = cropped_imgs[0, :3].transpose(1, 2, 0)
    # left_im = (((left_im - left_im.min()) / (left_im.max() - left_im.min()) * 1.) * 255).astype(np.uint8)
    # im = left_im[:, :, ::-1]
    # cv2.imwrite('centercrop.png', im)

    # right = imgs[0, :3].transpose(1, 2, 0)
    # right_im = (((right - right.min()) / (right.max() - right.min()) * 1.) * 255).astype(np.uint8)
    # im = right_im[:, :, ::-1]
    # cv2.imwrite('raw.png', im)


    return cropped_imgs


def random_crop_2(imgs, output_size):

    n1 = imgs.shape[0]
    n2 = imgs.shape[1]
    n = n1 * n2
    imgs = imgs.reshape(n, *imgs.shape[2:])
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0, :, :, 0]
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs.reshape(n1, n2, *cropped_imgs.shape[1:])



def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, top:top + new_h, left:left + new_w]
    return image


def conv2d_output_shape(h, w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Returns output H, W after convolution/pooling on input H, W.
    """
    kh, kw = kernel_size if isinstance(kernel_size,
                                       tuple) else (kernel_size, ) * 2
    sh, sw = stride if isinstance(stride, tuple) else (stride, ) * 2
    ph, pw = padding if isinstance(padding, tuple) else (padding, ) * 2
    d = dilation
    h = (h + (2 * ph) - (d * (kh - 1)) - 1) // sh + 1
    w = (w + (2 * pw) - (d * (kw - 1)) - 1) // sw + 1
    return h, w


class ScaleGrad(torch.autograd.Function):
    """Model component to scale gradients back from layer, without affecting
    the forward pass.  Used e.g. in dueling heads DQN models."""
    @staticmethod
    def forward(ctx, tensor, scale):
        """Stores the ``scale`` input to ``ctx`` for application in
        ``backward()``; simply returns the input ``tensor``."""
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Return the ``grad_output`` multiplied by ``ctx.scale``.  Also returns
        a ``None`` as placeholder corresponding to (non-existent) gradient of 
        the input ``scale`` of ``forward()``."""
        return grad_output * ctx.scale, None


# scale_grad = ScaleGrad.apply
# Supply a dummy for documentation to render.
scale_grad = getattr(ScaleGrad, "apply", None)


def update_state_dict(model, state_dict, tau=1, strip_ddp=True):
    """Update the state dict of ``model`` using the input ``state_dict``, which
    must match format.  ``tau==1`` applies hard update, copying the values, ``0<tau<1``
    applies soft update: ``tau * new + (1 - tau) * old``.
    """
    if strip_ddp:
        state_dict = strip_ddp_state_dict(state_dict)
    if tau == 1:
        model.load_state_dict(state_dict)
    elif tau > 0:
        update_sd = {
            k: tau * state_dict[k] + (1 - tau) * v
            for k, v in model.state_dict().items()
        }
        model.load_state_dict(update_sd)


def strip_ddp_state_dict(state_dict):
    """ Workaround the fact that DistributedDataParallel prepends 'module.' to
    every key, but the sampler models will not be wrapped in
    DistributedDataParallel. (Solution from PyTorch forums.)"""
    clean_state_dict = type(state_dict)()
    for k, v in state_dict.items():
        key = k[7:] if k[:7] == "module." else k
        clean_state_dict[key] = v
    return clean_state_dict
