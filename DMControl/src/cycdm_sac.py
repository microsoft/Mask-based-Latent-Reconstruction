# --------------------------------------------------------
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import (CenterCrop, RandomAffine, RandomCrop,
                                 RandomResizedCrop)
from kornia.filters import GaussianBlur2d

import utils
from encoder import make_encoder
from transition_model import make_transition_model

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_type,
                 encoder_feature_dim, log_std_min, log_std_max, num_layers,
                 num_filters):
        super().__init__()

        self.encoder = make_encoder(encoder_type,
                                    obs_shape,
                                    encoder_feature_dim,
                                    num_layers,
                                    num_filters,
                                    output_logits=True)
        print(self.encoder)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0]))

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self,
                obs,
                compute_pi=True,
                compute_log_pi=True,
                detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max -
                                            self.log_std_min) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(obs_dim + action_dim,
                                             hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_type,
                 encoder_feature_dim, num_layers, num_filters):
        super().__init__()

        self.encoder = make_encoder(encoder_type,
                                    obs_shape,
                                    encoder_feature_dim,
                                    num_layers,
                                    num_filters,
                                    output_logits=True)

        self.Q1 = QFunction(self.encoder.feature_dim, action_shape[0],
                            hidden_dim)
        self.Q2 = QFunction(self.encoder.feature_dim, action_shape[0],
                            hidden_dim)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


class CycDM(nn.Module):
    """Some Information about CycDM"""
    def __init__(self,
                 action_shape,
                 critic,
                 critic_target,
                 augmentation,
                 transition_model_type='probabilistic',
                 transition_model_layer_width=512,
                 encoder_feature_dim=50,
                 jumps=5,
                 latent_dim=512,
                 action_aug_type='random',
                 num_aug_actions=2,
                 loss_space='y',
                 bp_mode='gt',
                 cycle_steps=5,
                 cycle_mode='fp+cycle',
                 fp_loss_weight=5.0,
                 bp_loss_weight=1.0,
                 rc_loss_weight=0.0,
                 vc_loss_weight=1.0,
                 reward_loss_weight=0.0,
                 time_offset=0,
                 momentum_tau=1.0,
                 aug_prob=1.0,
                 sigma=0.05,
                 output_type="continuous"):
        super(CycDM, self).__init__()
        self.sigma = sigma
        self.jumps = jumps  # K
        self.loss_space = loss_space
        self.bp_mode = bp_mode
        self.cycle_steps = cycle_steps
        self.cycle_mode = cycle_mode
        self.fp_loss_weight = fp_loss_weight * self.jumps   # * K (as we mean it across time later) 
        self.bp_loss_weight = bp_loss_weight * self.jumps
        self.rc_loss_weight = rc_loss_weight
        self.vc_loss_weight = vc_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.time_offset = time_offset
        self.momentum_tau = momentum_tau
        self.aug_prob = aug_prob
        self.action_aug_type = action_aug_type
        self.num_aug_actions = num_aug_actions
        self.transition_model = make_transition_model(
            transition_model_type, encoder_feature_dim, action_shape,
            transition_model_layer_width)
        self.reverse_transition_model = make_transition_model(
            transition_model_type, encoder_feature_dim, action_shape,
            transition_model_layer_width)
        self.encoder = critic.encoder
        self.target_encoder = copy.deepcopy(critic.encoder)

        # original
        self.global_classifier = nn.Sequential(
            nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, encoder_feature_dim))
        self.global_target_classifier = nn.Sequential(
            nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, encoder_feature_dim))
        self.global_final_classifier = nn.Sequential(
            nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, encoder_feature_dim))

        self.transforms = []
        self.eval_transforms = []
        self.uses_augmentation = True
        for aug in augmentation:
            if aug == "affine":
                transformation = RandomAffine(5, (.14, .14), (.9, 1.1),
                                              (-5, 5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "crop":
                transformation = RandomCrop((84, 84))
                # Crashes if aug-prob not 1: use CenterCrop((84, 84)) or Resize((84, 84)) in that case.
                eval_transformation = CenterCrop((84, 84))
                self.uses_augmentation = True
                imagesize = 84
            elif aug == "rrc":
                transformation = RandomResizedCrop((100, 100), (0.8, 1))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "blur":
                transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "shift":
                transformation = nn.Sequential(nn.ReplicationPad2d(4),
                                               RandomCrop((84, 84)))
                eval_transformation = nn.Identity()
            elif aug == "intensity":
                transformation = Intensity(scale=0.05)
                eval_transformation = nn.Identity()
            elif aug == "none":
                transformation = eval_transformation = nn.Identity()
            else:
                raise NotImplementedError()
            self.transforms.append(transformation)
            self.eval_transforms.append(eval_transformation)

        self.reward_predictor = nn.Sequential(
            nn.Linear(encoder_feature_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 1))
        self.loss_fn = {
            'y': self.yspace_loss,
            'ygrad': self.yspace_loss,
            'z': self.zspace_loss,
            'mse': self.mse_loss,
            'spr': self.spr_loss
        }

    def apply_transforms(self, transforms, eval_transforms, image):
        if eval_transforms is None:
            for transform in transforms:
                image = transform(image)
        else:
            for transform, eval_transform in zip(transforms, eval_transforms):
                image = utils.maybe_transform(image,
                                              transform,
                                              eval_transform,
                                              p=self.aug_prob)
        return image

    @torch.no_grad()
    def transform(self, images, augment=False):
        images = images.float(
        ) / 255. if images.dtype == torch.uint8 else images
        flat_images = images.reshape(-1, *images.shape[-3:])
        if augment:
            processed_images = self.apply_transforms(self.transforms,
                                                     self.eval_transforms,
                                                     flat_images)
        else:
            processed_images = self.apply_transforms(self.eval_transforms,
                                                     None, flat_images)
        processed_images = processed_images.view(*images.shape[:-3],
                                                 *processed_images.shape[1:])
        return processed_images

    def norm_mse_loss(self, f_x1s, f_x2s, mean=True):
        f_x1 = F.normalize(f_x1s.float(), p=2., dim=-1,
                           eps=1e-3)  # (bs*(1+jumps), 512)
        f_x2 = F.normalize(f_x2s.float(), p=2., dim=-1, eps=1e-3)
        loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1)
        loss = loss.mean(0) if mean else loss
        return loss

    def spr_loss(self, latents, target_latents, observation, no_grad=False):
        if no_grad:
            with torch.no_grad():
                global_latents = self.global_classifier(latents)  # proj
                global_latents = self.global_final_classifier(
                    global_latents)  # pred
        else:
            global_latents = self.global_classifier(latents)  # proj
            global_latents = self.global_final_classifier(
                global_latents)  # pred

        with torch.no_grad():
            global_targets = self.global_target_classifier(target_latents)
        targets = global_targets.view(-1, observation.shape[1], self.jumps + 1,
                                      global_targets.shape[-1]).transpose(
                                          1, 2)
        latents = global_latents.view(-1, observation.shape[1], self.jumps + 1,
                                      global_latents.shape[-1]).transpose(
                                          1, 2)
        loss = self.norm_mse_loss(latents, targets, mean=False)
        # split to [jumps, bs]
        return loss.view(-1, observation.shape[1])

    def yspace_loss(self, latents, target_latents, no_grad=False):
        if no_grad:
            with torch.no_grad():
                global_latents = self.global_classifier(latents)  # proj
                global_latents = self.global_final_classifier(
                    global_latents)  # pred
        else:
            global_latents = self.global_classifier(latents)
            global_latents = self.global_final_classifier(global_latents)
        with torch.no_grad():
            global_targets = self.global_target_classifier(target_latents)
        '''
        targets = global_targets.view(-1, batch_size,
                                             timesteps, global_targets.shape[-1]).transpose(1, 2)
        latents = global_latents.view(-1, batch_size,
                                             timesteps, global_latents.shape[-1]).transpose(1, 2)
        loss = self.cycdm_loss(latents, targets)
        '''
        loss = self.norm_mse_loss(global_latents, global_targets, mean=False)
        # loss = loss.view(timesteps, batch_size) # (timesteps, bs)
        return loss

    def zspace_loss(self, latents, target_latents):
        # print(latents.size()) # (bs*M, 50)
        latents = F.normalize(latents, p=2., dim=-1, eps=1e-3)  # ((1+jumps)*bs, C, H*W)
        target_latents = F.normalize(target_latents, p=2., dim=-1, eps=1e-3)  # ((1+jumps)*bs, C, H*W)
        loss = F.mse_loss(latents, target_latents.clone().detach())
        # loss = torch.mean(F.mse_loss(latents, target_latents.detach(), reduction='none'), dim=[1,2,3]).view(timesteps, batch_size)
        return loss

    def mse_loss(self, latents, target_latents, observation, no_grad=None):
        loss = torch.mean(F.mse_loss(latents,
                                     target_latents.clone().detach(),
                                     reduction='none'),
                          dim=[1, 2, 3])
        return loss.view(-1, observation.shape[1])

    def do_forward_predict(self, first_latent, target_latents, observation,
                           action, reward):
        forward_pred_latents = [first_latent]
        forward_pred_rewards = []
        latent = first_latent
        pred_rew = self.reward_predictor(forward_pred_latents[0])
        forward_pred_rewards.append(F.log_softmax(pred_rew, -1))
        for j in range(1, self.jumps + 1):
            latent = self.transition_model.sample_prediction(
                torch.cat([latent, action[j - 1]], dim=1))
            forward_pred_latents.append(latent)
            pred_rew = self.reward_predictor(latent)[:action.shape[1]]
            forward_pred_rewards.append(F.log_softmax(pred_rew, -1))
        forward_pred_latents = torch.stack(forward_pred_latents, 1)
        forward_prediction_loss = self.loss_fn['spr'](
            forward_pred_latents.flatten(0, 1), target_latents, observation)
        forward_pred_rewards = torch.stack(forward_pred_rewards, 0)
        with torch.no_grad():
            reward_targets = utils.to_categorical(
                reward[:self.jumps + 1].flatten(),
                limit=0).view(*forward_pred_rewards.shape)
            reward_loss = -torch.sum(reward_targets * forward_pred_rewards,
                                     2).mean(0)
        return forward_prediction_loss[1:].mean(), reward_loss.mean(), latent

    def do_backward_predict(
        self,
        last_latent,
        target_latents,
        observation,
        action,
    ):
        if self.bp_mode == 'gt':
            backward_obs = observation[self.jumps].flatten(1, 2)
            backward_obs = self.transform(backward_obs, augment=True)
            _, T, B, img_shape = utils.infer_leading_dims(backward_obs, 3)
            backward_latent = self.encoder(backward_obs.view(
                T * B, *img_shape))

        elif self.bp_mode == 'detach':
            with torch.no_grad():
                backward_obs = observation[self.jumps].flatten(1, 2)
                backward_obs = self.transform(backward_obs, augment=True)
                _, T, B, img_shape = utils.infer_leading_dims(backward_obs, 3)
                backward_latent = self.encoder(
                    backward_obs.view(T * B, *img_shape))

        elif self.bp_mode == 'esti':
            backward_latent = last_latent
        elif self.bp_mode == 'hybrid':
            if torch.rand(1) > 0.5:
                backward_latent = cur_latents
            else:
                backward_obs = observation[self.jumps].flatten(1, 2)
                backward_obs = self.transform(backward_obs, augment=True)
                _, T, B, img_shape = utils.infer_leading_dims(backward_obs, 3)
                backward_latent = self.encoder(
                    backward_obs.view(T * B, *img_shape))
        latent = backward_latent
        backward_pred_latents = [latent]
        for j in reversed(range(1, 1 + self.jumps)):
            latent = self.reverse_transition_model.sample_prediction(
                torch.cat([latent, action[j - 1]], dim=1))
            backward_pred_latents.insert(0, latent)
        backward_pred_latents = torch.stack(backward_pred_latents,
                                            1).flatten(0, 1)
        backward_prediction_loss = self.loss_fn['spr'](
            backward_pred_latents,
            target_latents,
            observation,
            no_grad=self.bp_mode == 'detach')
        return backward_prediction_loss[:-1].mean()

    def sample_cycle_actions(self, aug_type, t, num_augs, prev_actions):
        if aug_type == 'random':
            actions = 2 * torch.rand(num_augs, *prev_actions.shape[1:], 
                device=prev_actions.device) - 1
            # noise = torch.randn(num_augs, *prev_actions.shape[1:], device=prev_actions.device)
            # actions = prev_actions[None, t] + noise * self.sigma
        elif aug_type == 'nonaug':
            actions = prev_actions[None, t]
        elif aug_type == 'hybrid':
            real_actions = prev_actions[None, t]
            noise = torch.randn(num_augs - 1, *prev_actions.shape[1:], 
                device=real_actions.device)
            aug_actions = real_actions + noise * self.sigma
            actions = torch.cat([real_actions, aug_actions], 0)
        actions = torch.clamp(actions, min=-1.0, max=1.0)
        return actions.detach()

    def do_cycle_predict(
        self,
        first_latent,
        target_latents,
        observation,
        action,
    ):
        shp = (self.num_aug_actions, action.size(1), *first_latent.shape[1:4])
        latent_t0 = first_latent[None].expand(*shp).flatten(0, 1)
        action_t0 = self.sample_cycle_actions(self.action_aug_type, 0,
                                              self.num_aug_actions,
                                              action).flatten(0, 1)
        action_t0 = action_t0.to(latent_t0.device)
        if self.loss_space == 'z':
            cycle_target = latent_t0.clone().detach()
        elif self.loss_space == 'y':
            cycle_target = observation[0].flatten(1, 2)
            
            # 1. Expand then Aug
            cycle_target = cycle_target[None].expand(
                self.num_aug_actions, *cycle_target.shape).flatten(0, 1)
            cycle_target = self.transform(cycle_target, augment=True)
            # # 2. Aug then expand
            # cycle_target = self.transform(cycle_target, augment=True)
            # cycle_target = cycle_target[None].expand(
            #     self.num_aug_actions, *cycle_target.shape).flatten(0, 1)
            
            with torch.no_grad():
                cycle_target = self.target_encoder(cycle_target)
            cycle_target = cycle_target.clone().detach()

            # # 3. No Aug
            # cycle_target = latent_t0.clone().detach()
        
        # forward prediction
        forward_pred_latents, forward_actions = [latent_t0], [action_t0]
        for i in range(1, 1 + self.cycle_steps):
            latent = self.transition_model.sample_prediction(
                torch.cat([forward_pred_latents[-1], forward_actions[-1]],
                          dim=1))
            forward_pred_latents.append(latent)
            if i < self.cycle_steps:
                synthetic_actions = self.sample_cycle_actions(
                    self.action_aug_type, i, self.num_aug_actions,
                    action).flatten(0, 1).to(latent.device)
                forward_actions.append(synthetic_actions)
        # backward prediction
        _ = forward_pred_latents.pop()
        total_virtual_cycle_loss = 0
        backward_pred_latents = [latent]
        for i in reversed(range(1, 1 + self.cycle_steps)):
            latent = self.reverse_transition_model.sample_prediction(
                torch.cat([backward_pred_latents[-1],
                           forward_actions.pop()],
                          dim=1))
            backward_pred_latents.append(latent)

            # # Every step
            # cycle_latent = latent
            # cycle_target = forward_pred_latents.pop().clone().detach()
            # if self.loss_space == 'z':
            #     # print(cycle_latent.size())  # (bs*M, 50)
            #     virtual_cycle_loss = self.zspace_loss(cycle_latent, cycle_target)
            # elif self.loss_space == 'y':
            #     virtual_cycle_loss = self.yspace_loss(cycle_latent,
            #                                         cycle_target,
            #                                         no_grad=False)
            #     virtual_cycle_loss = virtual_cycle_loss.view(self.num_aug_actions,
            #                                                 action.size(1)).mean()
            # total_virtual_cycle_loss = total_virtual_cycle_loss + virtual_cycle_loss

        # End step
        cycle_latent = latent
        if self.loss_space == 'z':
            # print(cycle_latent.size())  # (bs*M, 50)
            virtual_cycle_loss = self.zspace_loss(cycle_latent, cycle_target)
        elif self.loss_space == 'y':
            virtual_cycle_loss = self.yspace_loss(cycle_latent,
                                                  cycle_target,
                                                  no_grad=False)
            virtual_cycle_loss = virtual_cycle_loss.view(self.num_aug_actions,
                                                        action.size(1)).mean()

        return virtual_cycle_loss
        # return total_virtual_cycle_loss

    def do_real_cycle_loss(
        self,
        first_latent,
        target_latents,
        observation,
        action,
    ):
        shp = (1, action.size(1), *first_latent.shape[1:4])
        latent_t0 = first_latent[None].expand(*shp).flatten(0, 1)
        # latent_t0 = first_latent[None].expand(*shp).flatten(0, 1).clone().detach()
        action_t0 = self.sample_cycle_actions(aug_type='nonaug', t=0,
                                              num_augs=0,
                                              prev_actions=action).flatten(0, 1)
        action_t0 = action_t0.to(latent_t0.device)
        
        # forward prediction
        # with torch.no_grad():
        forward_pred_latents, forward_actions = [latent_t0], [action_t0]
        for i in range(1, 1 + self.cycle_steps):
            latent = self.transition_model.sample_prediction(
                torch.cat([forward_pred_latents[-1], forward_actions[-1]],
                        dim=1))
            forward_pred_latents.append(latent)
            if i < self.cycle_steps:
                synthetic_actions = self.sample_cycle_actions(aug_type='nonaug', t=i,
                    num_augs=0, prev_actions=action).flatten(0, 1).to(latent.device)
                forward_actions.append(synthetic_actions)
        
        # backward prediction
        rdm_loss = []
        backward_pred_latents = [latent]
        for i in reversed(range(1, 1 + self.cycle_steps)):
            latent = self.reverse_transition_model.sample_prediction(
                torch.cat([backward_pred_latents[-1],
                           forward_actions.pop()],
                          dim=1))
            backward_pred_latents.append(latent)
            cycle_latent = latent

            # if self.loss_space == 'z':
            #     cycle_target = latent_t0.clone().detach()
            # elif self.loss_space == 'y':
            cycle_target = observation[i-1].flatten(1, 2)
            cycle_target = cycle_target[None].expand(
                1, *cycle_target.shape).flatten(0, 1)
            cycle_target = self.transform(cycle_target, augment=True)
            with torch.no_grad():
                cycle_target = self.target_encoder(cycle_target)
            cycle_target = cycle_target.clone().detach()


            if self.loss_space == 'z':
                real_cycle_loss = F.mse_loss(cycle_latent,
                                                cycle_target,
                                                reduction='none').mean()
            elif self.loss_space == 'y':
                real_cycle_loss = self.yspace_loss(cycle_latent,
                                                    cycle_target,
                                                    # no_grad=True)
                                                      no_grad=False)
                real_cycle_loss = real_cycle_loss.view(1, action.size(1)).mean()
            rdm_loss.append(real_cycle_loss)
        # loss = torch.as_tensor(rdm_loss).mean()
        loss = rdm_loss[-1]
        return loss

    def do_cycdm_loss(self, first_latent, target_latents, observation, action,
                      reward, step):
        if 'fp' in self.cycle_mode:
            # forward prediction
            elements = self.do_forward_predict(first_latent, target_latents,
                                               observation, action, reward)
            forward_prediction_loss, reward_loss, last_latent = elements
        else:
            forward_prediction_loss = 0.0
            reward_loss = 0.0
        if 'bp' in self.cycle_mode:
            assert 'fp' in self.cycle_mode
            # backward prediction
            backward_prediction_loss = self.do_backward_predict(
                last_latent, target_latents, observation, action)
        else:
            backward_prediction_loss = 0.0
        if 'cycle' in self.cycle_mode:
            if self.vc_loss_weight == 0:
                virtual_cycle_loss = 0.
            else:
                virtual_cycle_loss = self.do_cycle_predict(first_latent,
                                                        target_latents,
                                                        observation, action)
            if self.rc_loss_weight == 0:
                real_cycle_loss = 0.
            else:
                real_cycle_loss = self.do_real_cycle_loss(first_latent,
                                                        target_latents,
                                                        observation, action)
        else:
            virtual_cycle_loss = 0.0
            real_cycle_loss = 0.0

        fp_loss = self.fp_loss_weight * forward_prediction_loss
        r_loss = self.reward_loss_weight * reward_loss
        bp_loss = self.bp_loss_weight * backward_prediction_loss
        # rc_loss = self.rc_loss_weight * real_cycle_loss
        # vc_loss = self.vc_loss_weight * virtual_cycle_loss
        warmup = 50000
        if step < warmup:
            warm_weight = np.exp(-5 * ((1 - (step+1)/warmup)**2))
            cur_vc_weight = self.vc_loss_weight * warm_weight
            cur_rc_weight = self.rc_loss_weight * warm_weight
        else:
            cur_vc_weight = self.vc_loss_weight
            cur_rc_weight = self.rc_loss_weight
        vc_loss = cur_vc_weight * virtual_cycle_loss
        rc_loss = cur_rc_weight * real_cycle_loss
        total_loss = fp_loss + bp_loss + rc_loss + vc_loss + r_loss
        losses_info = {
            'fp_loss': fp_loss,
            'bp_loss': bp_loss,
            'r_loss': r_loss,
            'rc_loss': rc_loss,
            'vc_loss': vc_loss
        }
        return total_loss, losses_info


class CycDMSacAgent(object):
    """CycDM representation learning with SAC."""
    def __init__(
            self,
            obs_shape,
            action_shape,
            device,
            # from CycDM
            augmentation=[],
            transition_model_type='probabilistic',
            transition_model_layer_width=512,
            jumps=5,
            latent_dim=512,
            time_offset=0,
            momentum_tau=1.0,
            aug_prob=1.0,
            auxiliary_task_lr=1e-3,
            action_aug_type='random',
            num_aug_actions=None,
            loss_space='y',
            bp_mode='gt',
            cycle_steps=5,
            cycle_mode='fp+cycle',
            fp_loss_weight=1.0,
            bp_loss_weight=1.0,
            rc_loss_weight=1.0,
            vc_loss_weight=1.0,
            reward_loss_weight=0.0,
            # from curl
            hidden_dim=256,
            discount=0.99,
            init_temperature=0.01,
            alpha_lr=1e-3,
            alpha_beta=0.9,
            actor_lr=1e-3,
            actor_beta=0.9,
            actor_log_std_min=-10,
            actor_log_std_max=2,
            actor_update_freq=2,
            critic_lr=1e-3,
            critic_beta=0.9,
            critic_tau=0.005,
            critic_target_update_freq=2,
            encoder_type='pixel',
            encoder_feature_dim=50,
            encoder_lr=1e-3,
            encoder_tau=0.005,
            num_layers=4,
            num_filters=32,
            cpc_update_freq=1,
            log_interval=100,
            detach_encoder=False,
            curl_latent_dim=128,
            sigma=0.05):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.curl_latent_dim = curl_latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type

        self.jumps = jumps
        self.reward_loss_weight = reward_loss_weight

        self.actor = Actor(obs_shape, action_shape, hidden_dim, encoder_type,
                           encoder_feature_dim, actor_log_std_min,
                           actor_log_std_max, num_layers,
                           num_filters).to(device)

        self.critic = Critic(obs_shape, action_shape, hidden_dim, encoder_type,
                             encoder_feature_dim, num_layers,
                             num_filters).to(device)

        self.critic_target = Critic(obs_shape, action_shape, hidden_dim,
                                    encoder_type, encoder_feature_dim,
                                    num_layers, num_filters).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=(actor_beta, 0.999))

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=(critic_beta, 0.999))

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=(alpha_beta, 0.999))

        if self.encoder_type == 'pixel':
            self.CycDM = CycDM(action_shape,
                               self.critic,
                               self.critic_target,
                               augmentation,
                               transition_model_type,
                               transition_model_layer_width,
                               encoder_feature_dim,
                               jumps,
                               latent_dim,
                               action_aug_type,
                               num_aug_actions,
                               loss_space,
                               bp_mode,
                               cycle_steps,
                               cycle_mode,
                               fp_loss_weight,
                               bp_loss_weight,
                               rc_loss_weight,
                               vc_loss_weight,
                               reward_loss_weight,
                               time_offset,
                               momentum_tau,
                               aug_prob,
                               sigma,
                               output_type='continuous').to(self.device)
            # optimizer for critic encoder and CycDM loss
            self.encoder_optimizer = torch.optim.Adam(
                # self.critic.encoder.parameters(), lr=encoder_lr)
                self.critic.encoder.parameters(), lr=0.5 * encoder_lr)
                # self.critic.encoder.parameters(), lr=2 * encoder_lr)
            self.cycdm_optimizer = torch.optim.Adam(
                self.CycDM.parameters(),
                lr=0.5 * auxiliary_task_lr
                # lr=auxiliary_task_lr)
                # [v for k,v in self.CycDM.named_parameters() if 'encoder' not in k],
                # lr=encoder_lr
                )

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type == 'pixel':
            self.CycDM.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(obs,
                                     compute_pi=False,
                                     compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)

        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_cycdm(self, cycdm_kwargs, L, step):
        # [1+self.jumps, B, 9, 1, 84, 84]
        observation = cycdm_kwargs["observation"]
        # [1+self.jumps, B, dim_A]
        action = cycdm_kwargs["action"]
        # [1+self.jumps, 1]
        reward = cycdm_kwargs["reward"]

        input_obs = observation[0].flatten(1, 2)
        input_obs = self.CycDM.transform(input_obs, augment=True)

        # stem forward
        lead_dim, T, B, img_shape = utils.infer_leading_dims(input_obs, 3)
        latent = self.CycDM.encoder(input_obs.view(
            T * B, *img_shape))  # Fold if T dimension.
        target_images = observation[self.CycDM.time_offset:self.CycDM.jumps +
                                    self.CycDM.time_offset + 1]
        target_images = target_images.transpose(0, 1).flatten(2, 3)
        target_images = self.CycDM.transform(target_images, augment=True)
        with torch.no_grad():
            target_latents = self.CycDM.target_encoder(
                target_images.flatten(0, 1))

        final_cycdm_loss, losses_info = self.CycDM.do_cycdm_loss(
            latent, target_latents, observation, action, reward, step)

        self.encoder_optimizer.zero_grad()
        self.cycdm_optimizer.zero_grad()
        final_cycdm_loss.backward()

        '''Oct 20, 2021: add grad norm'''
        model_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.encoder.parameters(), 10)
        model_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.CycDM.parameters(), 10)

        self.encoder_optimizer.step()
        self.cycdm_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train/cycdm_loss', final_cycdm_loss, step)
            L.log('train/forwad_prediction_loss', losses_info['fp_loss'], step)
            L.log('train/backward_prediction_loss', losses_info['bp_loss'],
                  step)
            L.log('train/real_cycle_loss', losses_info['rc_loss'], step)
            L.log('train/vitual_cycle_loss', losses_info['vc_loss'], step)
            L.log('train/reward_loss', losses_info['r_loss'], step)

    def update(self, replay_buffer, L, step):
        if self.encoder_type == 'pixel':
            elements = replay_buffer.sample_spr()
            obs, action, reward, next_obs, not_done, cycdm_kwargs = elements
        else:
            elements = replay_buffer.sample_proprio()
            obs, action, reward, next_obs, not_done = elements

        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        self.update_cycdm(cycdm_kwargs, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic.Q1, self.critic_target.Q1,
                                     self.critic_tau)
            utils.soft_update_params(self.critic.Q2, self.critic_target.Q2,
                                     self.critic_tau)
            utils.soft_update_params(self.critic.encoder,
                                     self.critic_target.encoder,
                                     self.encoder_tau)
            utils.soft_update_params(self.CycDM.encoder,
                                     self.CycDM.target_encoder,
                                     self.CycDM.momentum_tau)
            utils.soft_update_params(self.CycDM.global_classifier,
                                     self.CycDM.global_target_classifier,
                                     self.CycDM.momentum_tau)

    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(),
                   '%s/actor_%s.pt' % (model_dir, step))
        torch.save(self.critic.state_dict(),
                   '%s/critic_%s.pt' % (model_dir, step))

    def save_cycdm(self, model_dir, step):
        torch.save(self.CycDM.state_dict(),
                   '%s/cycdm_%s.pt' % (model_dir, step))

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step)))
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step)))
