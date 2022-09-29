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

# class InverseTransitionModel(nn.Module):
#     def __init__(self, action_dim, encoder_feature_dim=50, hidden_size=128):
#         super().__init__()
#         self.idm = nn.Sequential(
#             nn.Linear(encoder_feature_dim*2, hidden_size), nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size), nn.ReLU(),
#             nn.Linear(hidden_size, action_dim)
#         )
#         self.train()

#     def forward(self, state, next_state):
#         ''' s_t, s_t+1 '''
#         return self.idm(torch.cat([state, next_state], 1))

class InverseTransitionModel(nn.Module):
    def __init__(self, action_dim, encoder_feature_dim=50, hidden_size=128):
        super().__init__()
        self.idm = nn.Sequential(
            nn.Linear(encoder_feature_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        self.train()

    def forward(self, state, next_state):
        ''' s_t, s_t+1 '''
        return self.idm(next_state - state)

class IDMSacAgent(object):
    def __init__(
            self,
            obs_shape,
            action_shape,
            device,
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
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.curl_latent_dim = curl_latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type

        self.jumps = jumps

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

        ''' SPR '''
        self.aug_prob = aug_prob
        self.time_offset = time_offset
        self.momentum_tau = momentum_tau
        self.target_encoder = copy.deepcopy(self.critic.encoder)
        self.transition_model = make_transition_model(
            transition_model_type, encoder_feature_dim, action_shape,
            transition_model_layer_width).to(self.device)
        self.global_classifier = nn.Sequential(
            nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, encoder_feature_dim)).to(self.device)
        self.global_target_classifier = nn.Sequential(
            nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, encoder_feature_dim)).to(self.device)
        self.global_final_classifier = nn.Sequential(
            nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, encoder_feature_dim)).to(self.device)
        self.spr_params = list(self.critic.encoder.parameters()) \
                        + list(self.transition_model.parameters()) \
                        + list(self.global_classifier.parameters()) \
                        + list(self.global_final_classifier.parameters())
        # batch size 128: lr=0.5 * auxiliary_task_lr; 256: auxiliary_task_lr.
        self.spr_optimizer = torch.optim.Adam(self.spr_params, lr=0.5 * auxiliary_task_lr)

        ''' IDM '''
        self.inverse_transition_model = InverseTransitionModel(
            action_dim=action_shape[0],
            encoder_feature_dim=encoder_feature_dim,
            hidden_size=1024
            ).to(self.device)
        self.idm_params = list(self.critic.encoder.parameters()) \
                        + list(self.inverse_transition_model.parameters())
        self.idm_optimizer = torch.optim.Adam(self.idm_params, lr=1.0 * auxiliary_task_lr)

        ''' Data augmentation '''
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

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.transition_model.train(training)
        self.global_classifier.train(training)
        self.global_final_classifier.train(training)
        self.inverse_transition_model.train(training)

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

    def update_spr(self, spr_kwargs, L, step):
        ''' update the encoder and SPR modules (excluding IDM) '''
        observation = spr_kwargs["observation"] # [1+self.jumps, B, 9, 1, 100, 100]
        action = spr_kwargs["action"]   # [1+self.jumps, B, dim_A]
        reward = spr_kwargs["reward"]   # [1+self.jumps, 1]

        input_obs = observation[0].flatten(1, 2)
        input_obs = self.transform(input_obs, augment=True)

        # stem forward
        lead_dim, T, B, img_shape = utils.infer_leading_dims(input_obs, 3)
        latent = self.critic.encoder(input_obs.view(
            T * B, *img_shape))  # Fold if T dimension.
        target_images = observation[self.time_offset:self.jumps +
                                    self.time_offset + 1]
        target_images = target_images.transpose(0, 1).flatten(2, 3)
        target_images = self.transform(target_images, augment=True)
        with torch.no_grad():
            target_latents = self.target_encoder(
                target_images.flatten(0, 1))

        forw_spr_loss, forw_idm_loss = self.do_spr_loss(
            latent, target_latents, observation, action, reward, step)

        warmup = 100000
        if step < warmup:
            warm_weight = np.exp(-5 * ((1 - (step)/warmup)**2))
        else:
            warm_weight = 1

        # loss = (forw_spr_loss + forw_idm_loss) * self.jumps
        # loss = forw_spr_loss * self.jumps + forw_idm_loss * warm_weight
        loss = forw_spr_loss * self.jumps

        self.spr_optimizer.zero_grad()
        loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.spr_params, 10)
        self.spr_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train/spr_loss', forw_spr_loss * self.jumps, step)
            # L.log('train/spr_idm_loss', forw_idm_loss * warm_weight, step)

    def update_idm(self, idm_kwargs, L, step):
        ''' update only IDM '''
        self.inverse_transition_model.train()
        observation = idm_kwargs["observation"] # [2, B, 9, 1, 100, 100]
        action = idm_kwargs["action"]   # [2, B, dim_A]
        reward = idm_kwargs["reward"]   # [2, 1]

        input_obs = observation[0].flatten(1, 2)
        input_obs = self.transform(input_obs, augment=True)
        next_obs = observation[1].flatten(1, 2)
        next_obs = self.transform(next_obs, augment=True)

        lead_dim, T, B, img_shape = utils.infer_leading_dims(input_obs, 3)
        with torch.no_grad():
            latent = self.target_encoder(input_obs.view(
                T * B, *img_shape))  # Fold if T dimension.

            next_latent = self.target_encoder(next_obs.view(
                T * B, *img_shape))  # Fold if T dimension.
        
        pred_action = self.inverse_transition_model(latent, next_latent)
        idm_loss = F.mse_loss(pred_action, action[0])

        self.idm_optimizer.zero_grad()
        idm_loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(self.idm_params, 10)
        self.idm_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train/idm_loss', idm_loss, step)
        
        self.inverse_transition_model.eval()

    def update(self, replay_buffer, L, step):
        if self.encoder_type == 'pixel':
            elements = replay_buffer.sample_spr()
            obs, action, reward, next_obs, not_done, spr_kwargs = elements
            idm_kwargs = replay_buffer.sample_idm()
        else:
            elements = replay_buffer.sample_proprio()
            obs, action, reward, next_obs, not_done = elements

        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)
        
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        
        # idm_update_freq = 1
        # for _ in range(idm_update_freq):
        #     self.update_idm(idm_kwargs, L, step)
        
        self.update_spr(spr_kwargs, L, step)

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
            utils.soft_update_params(self.critic.encoder,
                                     self.target_encoder,
                                     self.momentum_tau)
            utils.soft_update_params(self.global_classifier,
                                     self.global_target_classifier,
                                     self.momentum_tau)

    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(),
                   '%s/actor_%s.pt' % (model_dir, step))
        torch.save(self.critic.state_dict(),
                   '%s/critic_%s.pt' % (model_dir, step))

    def save_cycdm(self, model_dir, step):
        pass

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step)))
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step)))

    def do_spr_loss(self, first_latent, target_latents, observation, action,
                      reward=None, step=None):
        self.inverse_transition_model.eval()
        idm_loss = 0
        forward_pred_latents = [first_latent]
        latent = first_latent
        for j in range(1, self.jumps + 1):
            previous_latent = latent.clone().detach()
            latent = self.transition_model.sample_prediction(
                torch.cat([latent, action[j - 1]], dim=1))
            forward_pred_latents.append(latent)
            pred_action = self.inverse_transition_model(previous_latent, latent)
            idm_loss = idm_loss + F.mse_loss(pred_action, action[j - 1].detach())
        forward_pred_latents = torch.stack(forward_pred_latents, 1)
        forward_prediction_loss = self.spr_loss(
            forward_pred_latents.flatten(0, 1), target_latents, observation)
        return forward_prediction_loss[1:].mean(), idm_loss

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

    def norm_mse_loss(self, f_x1s, f_x2s, mean=True):
        f_x1 = F.normalize(f_x1s.float(), p=2., dim=-1,
                           eps=1e-3)  # (bs*(1+jumps), 512)
        f_x2 = F.normalize(f_x2s.float(), p=2., dim=-1, eps=1e-3)
        loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1)
        loss = loss.mean(0) if mean else loss
        return loss
