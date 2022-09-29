# --------------------------------------------------------
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import copy
import torch
import torch.nn.functional as F
import torch.nn as nn

from rlpyt.models.utils import scale_grad, update_state_dict
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from src.utils import count_parameters, dummy_context_mgr
import numpy as np
from kornia.augmentation import RandomAffine,\
    RandomCrop,\
    CenterCrop, \
    RandomResizedCrop
from kornia.filters import GaussianBlur2d
import copy
import wandb
import math

from src.vit_modules import Block, trunc_normal_

from src.masking_generator import CubeMaskGenerator


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

class PVCatDqnModel(torch.nn.Module):
    """2D conlutional network feeding into MLP with ``n_atoms`` outputs
    per action, representing a discrete probability distribution of Q-values."""

    def __init__(
            self,
            image_shape,
            output_size,
            n_atoms,
            dueling,
            jumps,
            spr,
            augmentation,
            target_augmentation,
            eval_augmentation,
            dynamics_blocks,
            norm_type,
            noisy_nets,
            aug_prob,
            classifier,
            imagesize,
            time_offset,
            local_spr,
            global_spr,
            momentum_encoder,
            shared_encoder,
            distributional,
            dqn_hidden_size,
            momentum_tau,
            renormalize,
            q_l1_type,
            dropout,
            final_classifier,
            model_rl,
            noisy_nets_std,
            residual_tm,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            framestack=4,
            cycle_step=2,
            space='y',
            real_cycle=True,
            virtual_cycle=True,
            aug_num=None,
            fp=False,
            bp=False,
            bp_mode='gt',
            aug_type='random',
    ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()

        self.cycle_step = cycle_step
        self.space = space
        self.real_cycle = real_cycle
        self.virtual_cycle = virtual_cycle
        self.forward_predict = fp
        self.backward_predict = bp
        self.bp_mode = bp_mode
        self.aug_type = aug_type
        self.aug_num = output_size if aug_num == None else int(aug_num * output_size)
        if self.aug_type == 'hybrid' and self.aug_num == 0:
            self.aug_num = 1

        self.noisy = noisy_nets
        self.time_offset = time_offset
        self.aug_prob = aug_prob
        self.classifier_type = classifier

        self.distributional = distributional
        n_atoms = 1 if not self.distributional else n_atoms
        self.dqn_hidden_size = dqn_hidden_size

        self.transforms = []
        self.eval_transforms = []

        self.uses_augmentation = False
        for aug in augmentation:
            if aug == "affine":
                transformation = RandomAffine(5, (.14, .14), (.9, 1.1), (-5, 5))
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
                transformation = nn.Sequential(nn.ReplicationPad2d(4), RandomCrop((84, 84)))
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

        self.dueling = dueling
        f, c = image_shape[:2]
        in_channels = np.prod(image_shape[:2])
        self.conv = Conv2dModel(
            in_channels=in_channels,
            channels=[32, 64, 64],
            kernel_sizes=[8, 4, 3],
            strides=[4, 2, 1],
            paddings=[0, 0, 0],
            use_maxpool=False,
            dropout=dropout,
        )

        fake_input = torch.zeros(1, f*c, imagesize, imagesize)
        fake_output = self.conv(fake_input)
        self.hidden_size = fake_output.shape[1]
        self.pixels = fake_output.shape[-1]*fake_output.shape[-2]
        print("Spatial latent size is {}".format(fake_output.shape[1:]))

        self.jumps = jumps
        # self.jumps = cycle_step
        self.model_rl = model_rl
        self.use_spr = spr
        self.target_augmentation = target_augmentation
        self.eval_augmentation = eval_augmentation
        self.num_actions = output_size

        if dueling:
            self.head = DQNDistributionalDuelingHeadModel(self.hidden_size,
                                                          output_size,
                                                          hidden_size=self.dqn_hidden_size,
                                                          pixels=self.pixels,
                                                          noisy=self.noisy,
                                                          n_atoms=n_atoms,
                                                          std_init=noisy_nets_std)
        else:
            self.head = DQNDistributionalHeadModel(self.hidden_size,
                                                   output_size,
                                                   hidden_size=self.dqn_hidden_size,
                                                   pixels=self.pixels,
                                                   noisy=self.noisy,
                                                   n_atoms=n_atoms,
                                                   std_init=noisy_nets_std)

        if self.jumps > 0:
            self.dynamics_model = TransitionModel(channels=self.hidden_size,
                                                  num_actions=output_size,
                                                  pixels=self.pixels,
                                                  hidden_size=self.hidden_size,
                                                  limit=1,
                                                  blocks=dynamics_blocks,
                                                  norm_type=norm_type,
                                                  renormalize=renormalize,
                                                  residual=residual_tm)
            self.backward_dynamics_model = BackwardTransitionModel(channels=self.hidden_size,
                                                num_actions=output_size,
                                                pixels=self.pixels,
                                                hidden_size=self.hidden_size,
                                                limit=1,
                                                blocks=dynamics_blocks,
                                                norm_type=norm_type,
                                                renormalize=renormalize,
                                                residual=residual_tm)
            # self.inverse_dynamics_model = InverseTransitionModel(channels=self.hidden_size,
            #                                       num_actions=output_size,
            #                                       pixels=self.pixels,
            #                                       hidden_size=self.hidden_size,
            #                                       limit=1,
            #                                       blocks=dynamics_blocks,
            #                                       norm_type=norm_type,
            #                                       renormalize=renormalize,
            #                                       residual=residual_tm)
        else:
            self.dynamics_model = nn.Identity()


        # Add a transformer
        num_attn_layers = 2
        encoder_feature_dim = 7*7*64
        num_heads = 1
        self.transformer = nn.ModuleList([
            Block(encoder_feature_dim, num_heads, mlp_ratio=2., 
                    qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                    drop_path=0., init_values=0., act_layer=nn.GELU, 
                    norm_layer=nn.LayerNorm, attn_head_dim=None) 
            for _ in range(num_attn_layers)])
        # self.action_emb = nn.Linear(action_shape[0], encoder_feature_dim)
        self.position = PositionalEmbedding(encoder_feature_dim)


        self.renormalize = renormalize

        if self.use_spr:
            self.local_spr = local_spr
            self.global_spr = global_spr
            self.momentum_encoder = momentum_encoder
            self.momentum_tau = momentum_tau
            self.shared_encoder = shared_encoder
            assert not (self.shared_encoder and self.momentum_encoder)

            # in case someone tries something silly like --local-spr 2
            self.num_sprs = int(bool(self.local_spr)) + \
                            int(bool(self.global_spr))

            if self.local_spr:
                self.local_final_classifier = nn.Identity()
                if self.classifier_type == "mlp":
                    self.local_classifier = nn.Sequential(nn.Linear(self.hidden_size,
                                                                    self.hidden_size),
                                                          nn.BatchNorm1d(self.hidden_size),
                                                          nn.ReLU(),
                                                          nn.Linear(self.hidden_size,
                                                                    self.hidden_size))
                elif self.classifier_type == "bilinear":
                    self.local_classifier = nn.Linear(self.hidden_size, self.hidden_size)
                elif self.classifier_type == "none":
                    self.local_classifier = nn.Identity()
                if final_classifier == "mlp":
                    self.local_final_classifier = nn.Sequential(nn.Linear(self.hidden_size, 2*self.hidden_size),
                                                                nn.BatchNorm1d(2*self.hidden_size),
                                                                nn.ReLU(),
                                                                nn.Linear(2*self.hidden_size,
                                                                    self.hidden_size))
                elif final_classifier == "linear":
                    self.local_final_classifier = nn.Linear(self.hidden_size, self.hidden_size)
                else:
                    self.local_final_classifier = nn.Identity()

                self.local_target_classifier = self.local_classifier
            else:
                self.local_classifier = self.local_target_classifier = nn.Identity()
            if self.global_spr:
                self.global_final_classifier = nn.Identity()
                if self.classifier_type == "mlp":
                    self.global_classifier = nn.Sequential(
                                                nn.Flatten(-3, -1),
                                                nn.Linear(self.pixels*self.hidden_size, 512),
                                                nn.BatchNorm1d(512),
                                                nn.ReLU(),
                                                nn.Linear(512, 256)
                                                )
                    self.global_target_classifier = self.global_classifier
                    global_spr_size = 256
                elif self.classifier_type == "q_l1":
                    self.global_classifier = QL1Head(self.head, dueling=dueling, type=q_l1_type)
                    global_spr_size = self.global_classifier.out_features
                    self.global_target_classifier = self.global_classifier
                elif self.classifier_type == "q_l2":
                    self.global_classifier = nn.Sequential(self.head, nn.Flatten(-2, -1))
                    self.global_target_classifier = self.global_classifier
                    global_spr_size = 256
                elif self.classifier_type == "bilinear":
                    self.global_classifier = nn.Sequential(nn.Flatten(-3, -1),
                                                           nn.Linear(self.hidden_size*self.pixels,
                                                                     self.hidden_size*self.pixels))
                    self.global_target_classifier = nn.Flatten(-3, -1)
                elif self.classifier_type == "none":
                    self.global_classifier = nn.Flatten(-3, -1)
                    self.global_target_classifier = nn.Flatten(-3, -1)

                    global_spr_size = self.hidden_size*self.pixels
                if final_classifier == "mlp":
                    self.global_final_classifier = nn.Sequential(
                        nn.Linear(global_spr_size, global_spr_size*2),
                        nn.BatchNorm1d(global_spr_size*2),
                        nn.ReLU(),
                        nn.Linear(global_spr_size*2, global_spr_size)
                    )
                elif final_classifier == "linear":
                    self.global_final_classifier = nn.Sequential(
                        nn.Linear(global_spr_size, global_spr_size),
                    )
                elif final_classifier == "none":
                    self.global_final_classifier = nn.Identity()
            else:
                self.global_classifier = self.global_target_classifier = nn.Identity()

            if self.momentum_encoder:
                self.target_encoder = copy.deepcopy(self.conv)
                self.global_target_classifier = copy.deepcopy(self.global_target_classifier)
                self.local_target_classifier = copy.deepcopy(self.local_target_classifier)
                for param in (list(self.target_encoder.parameters())
                            + list(self.global_target_classifier.parameters())
                            + list(self.local_target_classifier.parameters())):
                    param.requires_grad = False

            elif not self.shared_encoder:
                # Use a separate target encoder on the last frame only.
                self.global_target_classifier = copy.deepcopy(self.global_target_classifier)
                self.local_target_classifier = copy.deepcopy(self.local_target_classifier)
                if self.stack_actions:
                    input_size = c - 1
                else:
                    input_size = c
                self.target_encoder = Conv2dModel(in_channels=input_size,
                                                  channels=[32, 64, 64],
                                                  kernel_sizes=[8, 4, 3],
                                                  strides=[4, 2, 1],
                                                  paddings=[0, 0, 0],
                                                  use_maxpool=False,
                                                  )

            elif self.shared_encoder:
                self.target_encoder = self.conv

        img_size = 84
        mask_ratio = 0.5
        block_size = 8
        patch_size = 12
        input_size = img_size // patch_size
        self.masker = CubeMaskGenerator(
            input_size=input_size, image_size=img_size, clip_size=self.jumps+1, \
                block_size=block_size, mask_ratio=mask_ratio)  # 1 for mask, num_grid=input_size
        
        self.action_embedding = nn.Linear(self.num_actions, encoder_feature_dim)

        print("Initialized model with {} parameters".format(count_parameters(self)))

    def set_sampling(self, sampling):
        if self.noisy:
            self.head.set_sampling(sampling)

    def spr_loss(self, f_x1s, f_x2s, mean=True):
        f_x1 = F.normalize(f_x1s.float(), p=2., dim=-1, eps=1e-3)   # (bs*(1+jumps), 512)
        f_x2 = F.normalize(f_x2s.float(), p=2., dim=-1, eps=1e-3)
        loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1)
        loss = loss.mean(0) if mean else loss
        return loss

    def cal_acc(self, logits, labels):
        preds = torch.max(logits, dim=1)[1]
        return torch.mean((preds == labels.float()).float())

    # def global_spr_loss(self, latents, target_latents, batch_size):
    #     global_latents = self.global_classifier(latents)    # proj
    #     global_latents = self.global_final_classifier(global_latents)   # pred
    #     with torch.no_grad() if self.momentum_encoder else dummy_context_mgr():
    #         global_targets = self.global_target_classifier(target_latents)
    #     targets = global_targets.view(-1, batch_size,
    #                                          self.jumps+1, global_targets.shape[-1]).transpose(1, 2)
    #     latents = global_latents.view(-1, batch_size,
    #                                          self.jumps+1, global_latents.shape[-1]).transpose(1, 2)
    #     loss = self.spr_loss(latents, targets)
    #     return loss

    def global_spr_loss(self, latents, target_latents, observation):
        global_latents = self.global_classifier(latents)
        global_latents = self.global_final_classifier(global_latents)
        with torch.no_grad() if self.momentum_encoder else dummy_context_mgr():
            global_targets = self.global_target_classifier(target_latents)
        targets = global_targets.view(-1, observation.shape[1],
                                             self.jumps+1, global_targets.shape[-1]).transpose(1, 2)
        latents = global_latents.view(-1, observation.shape[1],
                                             self.jumps+1, global_latents.shape[-1]).transpose(1, 2)
        # print(latents.size(), targets.size())   # [1, 7, 32, 512]
        loss = self.spr_loss(latents, targets)
        # print(loss.size())  # [7, 32]
        return loss

    def local_spr_loss(self, latents, target_latents, observation):
        local_latents = latents.flatten(-2, -1).permute(2, 0, 1)
        local_latents = self.local_classifier(local_latents)
        local_latents = self.local_final_classifier(local_latents)
        local_target_latents = target_latents.flatten(-2, -1).permute(2, 0, 1)
        with torch.no_grad() if self.momentum_encoder else dummy_context_mgr():
            local_targets = self.local_target_classifier(local_target_latents)

        local_latents = local_latents.view(-1,
                                           observation.shape[1],
                                           self.jumps+1,
                                           local_latents.shape[-1]).transpose(1, 2)
        local_targets = local_targets.view(-1,
                                           observation.shape[1],
                                           self.jumps+1,
                                           local_targets.shape[-1]).transpose(1, 2)
        local_loss = self.spr_loss(local_latents, local_targets)
        return local_loss

    def do_spr_loss(self, pred_latents, observation):
        pred_latents = torch.stack(pred_latents, 1)
        latents = pred_latents[:observation.shape[1]].flatten(0, 1)  # batch*jumps, *
        neg_latents = pred_latents[observation.shape[1]:].flatten(0, 1)
        latents = torch.cat([latents, neg_latents], 0)
        target_images = observation[self.time_offset:\
            self.jumps + self.time_offset+1].transpose(0, 1).flatten(2, 3)
        target_images = self.transform(target_images, True)

        if not self.momentum_encoder and not self.shared_encoder:
            target_images = target_images[..., -1:, :, :]
        with torch.no_grad() if self.momentum_encoder else dummy_context_mgr():
            target_latents = self.target_encoder(target_images.flatten(0, 1))
            if self.renormalize:
                target_latents = renormalize(target_latents, -3)

        if self.local_spr:
            local_loss = self.local_spr_loss(latents, target_latents, observation)
        else:
            local_loss = 0
        if self.global_spr:
            global_loss = self.global_spr_loss(latents, target_latents, observation)
        else:
            global_loss = 0

        spr_loss = (global_loss + local_loss)/self.num_sprs
        spr_loss = spr_loss.view(-1, observation.shape[1]) # split to batch, jumps

        if self.momentum_encoder:
            update_state_dict(self.target_encoder,
                              self.conv.state_dict(),
                              self.momentum_tau)
            if self.classifier_type != "bilinear":
                # q_l1 is also bilinear for local
                if self.local_spr and self.classifier_type != "q_l1":
                    update_state_dict(self.local_target_classifier,
                                      self.local_classifier.state_dict(),
                                      self.momentum_tau)
                if self.global_spr:
                    update_state_dict(self.global_target_classifier,
                                      self.global_classifier.state_dict(),
                                      self.momentum_tau)
        return spr_loss

    def yspace_loss(self, latents, target_latents, no_grad=False):
        if no_grad:
            with torch.no_grad():
                global_latents = self.global_classifier(latents)    # proj
                global_latents = self.global_final_classifier(global_latents)   # pred
        else:
            global_latents = self.global_classifier(latents)
            global_latents = self.global_final_classifier(global_latents)
        with torch.no_grad() if self.momentum_encoder else dummy_context_mgr():
            global_targets = self.global_target_classifier(target_latents)
        '''
        targets = global_targets.view(-1, batch_size,
                                             timesteps, global_targets.shape[-1]).transpose(1, 2)
        latents = global_latents.view(-1, batch_size,
                                             timesteps, global_latents.shape[-1]).transpose(1, 2)
        loss = self.spr_loss(latents, targets)
        '''
        loss = self.spr_loss(global_latents, global_targets, mean=False)
        # loss = loss.view(timesteps, batch_size) # (timesteps, bs)
        return loss

    def zspace_loss(self, latents, target_latents, batch_size, timesteps):
        latents = F.normalize(latents.flatten(-2, -1), p=2., dim=-1, eps=1e-3)  # ((1+jumps)*bs, C, H*W)
        target_latents = F.normalize(target_latents.flatten(-2, -1), p=2., dim=-1, eps=1e-3)  # ((1+jumps)*bs, C, H*W)
        loss = F.mse_loss(latents, target_latents.clone().detach(), 
                reduction='none').sum(-1).mean(-1).view(timesteps, batch_size)
        # loss = torch.mean(F.mse_loss(latents, target_latents.detach(), reduction='none'), dim=[1,2,3]).view(timesteps, batch_size)
        return loss # (timesteps, bs)

    def mse_loss(self, latents, target_latents, batch_size, timesteps):
        # latents = F.normalize(latents.flatten(-2, -1), p=2., dim=-1, eps=1e-3)  # ((1+jumps)*bs, C, H*W)
        # target_latents = F.normalize(target_latents.flatten(-2, -1), p=2., dim=-1, eps=1e-3)  # ((1+jumps)*bs, C, H*W)
        # loss = F.mse_loss(latents, target_latents.detach(), reduction='none').sum(-1).mean(-1).view(timesteps, batch_size)
        loss = torch.mean(F.mse_loss(latents, target_latents.clone().detach(), 
                reduction='none'), dim=[1,2,3]).view(timesteps, batch_size)
        return loss # (timesteps, bs)

    def my_spr_loss(self, latents, target_latents, batch_size, timesteps, no_grad=False):
        if no_grad:
            with torch.no_grad():
                global_latents = self.global_classifier(latents)    # proj
                global_latents = self.global_final_classifier(global_latents)   # pred
        else:
            global_latents = self.global_classifier(latents)    # proj
            global_latents = self.global_final_classifier(global_latents)   # pred
        with torch.no_grad() if self.momentum_encoder else dummy_context_mgr():
            global_targets = self.global_target_classifier(target_latents)
        targets = global_targets.view(-1, batch_size,
                                             timesteps, global_targets.shape[-1]).transpose(1, 2)
        latents = global_latents.view(-1, batch_size,
                                             timesteps, global_latents.shape[-1]).transpose(1, 2)
        # print(latents.size())   # [1, 7, 32, 512]
        loss = self.spr_loss(latents, targets)
        # print(loss.size())  # [7, 32]
        return loss

    def prediction_loss(self, forward_latents, backward_latents, target_latents, space, batch_size, timesteps):
        if space == 'y' or space == 'ygrad':
            sim_loss = self.yspace_loss
        elif space == 'z':
            sim_loss = self.zspace_loss
        elif space == 'mse':
            sim_loss = self.mse_loss
        else:
            raise NameError
        forward_loss = sim_loss(forward_latents, target_latents, batch_size, timesteps)
        backward_loss = sim_loss(backward_latents, target_latents, batch_size, timesteps)
        return forward_loss, backward_loss

    def byol_loss(self, latents, target_latents, space, batch_size, timesteps, no_grad=False):
        if space == 'y' or space == 'ygrad':
            sim_loss = self.yspace_loss
        elif space == 'z':
            sim_loss = self.zspace_loss
        elif space == 'mse':
            sim_loss = self.mse_loss
        elif space == 'spr':
            sim_loss = self.my_spr_loss
            loss = sim_loss(latents, target_latents, batch_size, timesteps, no_grad)
            return loss
        else:
            raise NameError
        loss = sim_loss(latents, target_latents, batch_size, timesteps)
        # print("return")
        return loss

    def momentum_update(self, ):
        if self.momentum_encoder:
            update_state_dict(self.target_encoder,
                              self.conv.state_dict(),
                              self.momentum_tau)
            if self.classifier_type != "bilinear":
                # q_l1 is also bilinear for local
                if self.local_spr and self.classifier_type != "q_l1":
                    update_state_dict(self.local_target_classifier,
                                      self.local_classifier.state_dict(),
                                      self.momentum_tau)
                if self.global_spr:
                    update_state_dict(self.global_target_classifier,
                                      self.global_classifier.state_dict(),
                                      self.momentum_tau)
    
    def apply_transforms(self, transforms, eval_transforms, image):
        if eval_transforms is None:
            for transform in transforms:
                image = transform(image)
        else:
            for transform, eval_transform in zip(transforms, eval_transforms):
                image = maybe_transform(image, transform,
                                        eval_transform, p=self.aug_prob)
        return image

    @torch.no_grad()
    def transform(self, images, augment=False):
        images = images.float()/255. if images.dtype == torch.uint8 else images
        flat_images = images.reshape(-1, *images.shape[-3:])
        if augment:
            processed_images = self.apply_transforms(self.transforms,
                                                     self.eval_transforms,
                                                     flat_images)
        else:
            processed_images = self.apply_transforms(self.eval_transforms,
                                                     None,
                                                     flat_images)
        processed_images = processed_images.view(*images.shape[:-3],
                                                 *processed_images.shape[1:])
        return processed_images

    def stem_parameters(self):
        return list(self.conv.parameters()) + list(self.head.parameters())

    def stem_forward(self, img, prev_action=None, prev_reward=None):
        """Returns the normalized output of convolutional layers."""
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
        if self.renormalize:
            conv_out = renormalize(conv_out, -3)
        return conv_out

    def head_forward(self,
                     conv_out,
                     prev_action,
                     prev_reward,
                     logits=False):
        lead_dim, T, B, img_shape = infer_leading_dims(conv_out, 3)
        p = self.head(conv_out)

        if self.distributional:
            if logits:
                p = F.log_softmax(p, dim=-1)
            else:
                p = F.softmax(p, dim=-1)
        else:
            p = p.squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        p = restore_leading_dims(p, lead_dim, T, B)
        return p

    def sample_cycle_actions(self, aug_type, t, aug_num, batch_size, num_actions, prev_action, device):
        if aug_type == 'random':
            actions = self.generate_random_actions(num_actions=num_actions, 
                shape=(aug_num, batch_size), device=device)
        elif aug_type == 'nonaug':
            actions = prev_action[t+1][None]
        elif aug_type == 'hybrid':
            real_actions = prev_action[t+1][None]
            aug_actions = self.generate_random_actions(num_actions=num_actions, 
                shape=(aug_num-1, batch_size), device=device)
            actions = torch.cat([real_actions, aug_actions], 0)
        else:
            raise NotImplementedError
        return actions.detach()

    def forward(self, observation,
                prev_action, prev_reward, 
                train=False, eval=False):
        """
        For convenience reasons with DistributedDataParallel the forward method
        has been split into two cases, one for training and one for eval.
        obsevation: (1+jumps+n_step, bs, 4, 1, 84, 84)
        prev_action: 1+jumps+n_step elements list, each elsement: (bs, ) 
        prev_reward: 1+jumps+n_step elements list, each elsement: (bs, ) 
        """
        if train:
            # prediction
            log_pred_ps = []
            pred_reward = []
            input_obs = observation[0].flatten(1, 2)
            input_obs = self.transform(input_obs, augment=True)
            latent = self.stem_forward(input_obs,
                                       prev_action[0],
                                       prev_reward[0])
            batch_size = latent.size(0)
            log_pred_ps.append(self.head_forward(latent,    # RL forward
                                                 prev_action[0],
                                                 prev_reward[0],
                                                 logits=True))
            forward_latents, forward_actions  = [latent], []
            pred_rew = self.dynamics_model.reward_predictor(latent)
            pred_reward.append(F.log_softmax(pred_rew, -1))

            # create mask
            mask = self.masker()    # [jumps+1, 1, 84, 84], T = jumps+1
            mask = mask[:, None, None].expand(mask.size(0), *observation.size()[1:]).to(latent.device)
            # print(mask.size())  # [T, B, 4, 1, 84, 84]

            # sample actions
            action = prev_action[1 : 2 + self.jumps]
            T, B = action.size()
            action = action.flatten(0, 1) # (T*B, )
            batch_range = torch.arange(action.shape[0], device=action.device)
            action_onehot = torch.zeros(action.shape[0],
                                        self.num_actions,
                                        device=action.device)
            action_onehot[batch_range, action] = 1
            action = self.action_embedding(action_onehot) #(T*B, A)
            action = action.view(T, B, -1).transpose(0, 1)   # (T, B, 64*7*7) -> (B, T, 7*7*64)
            

            # sample targets
            max_jumps = self.jumps
            target_images = observation[:max_jumps + 1]\
                .transpose(0, 1).flatten(2, 3)  # (T,bs,4,1,84,84) -> (bs,T,4,1,84,84) -> (bs,T,4,84,84)
            target_images = self.transform(target_images, True) # (bs,T,4,84,84)
            if not self.momentum_encoder and not self.shared_encoder:
                target_images = target_images[..., -1:, :, :]   # (bs,T,4,84,84)
            with torch.no_grad() if self.momentum_encoder else dummy_context_mgr():
                target_latents = self.target_encoder(target_images.flatten(0, 1))   # (bs*T,4,84,84) -> (bs*T,64,7,7)
                if self.renormalize:
                    target_latents = renormalize(target_latents, -3)    # (bs*T,64,7,7)

            # masking
            # print((observation[:max_jumps + 1]).size(), mask.size())
            masked_images = (observation[:max_jumps + 1] * mask).transpose(0, 1).flatten(2, 3)  # (T,bs,4,1,84,84) -> (bs,T,4,1,84,84) -> (bs,T,4,84,84)
            masked_images = self.transform(masked_images, True) # (bs,T,4,84,84)
            B, T, C, H, W = masked_images.size()
            masked_images = masked_images.flatten(0, 1)
            masked_latents = self.stem_forward(masked_images) # (B*T,64,7,7)
            c, h, w = masked_latents.size()[-3:]
            masked_latents = masked_latents.view(B, T, c, h, w)

            # Decoding
            masked_latents = masked_latents.view(B, T, -1)
            # positional emb
            position = self.position(T).to(mask.device)
            position = position.expand(B, T, -1)
            masked_latents = masked_latents + position
            action = action + position
            x_full = torch.cat([masked_latents, action], 1)
            # action emb
            actions = prev_action
            for i in range(len(self.transformer)):
                x_full = self.transformer[i](x_full)    # (B, 2*T, 64*7*7)
            masked_latents = x_full[:, :T]
            masked_latents = masked_latents.view(B, T, c, h, w)
            masked_latents = masked_latents.flatten(0, 1)

            mlr_loss = self.byol_loss(
                            masked_latents, 
                            target_latents, 
                            'spr', 
                            observation.size(1), 
                            1+self.jumps)

            self.momentum_update()
            return log_pred_ps, pred_reward, mlr_loss

        else:
            aug_factor = self.target_augmentation if not eval else self.eval_augmentation
            observation = observation.flatten(-4, -3)
            stacked_observation = observation.unsqueeze(1).repeat(1, max(1, aug_factor), 1, 1, 1)
            stacked_observation = stacked_observation.view(-1, *observation.shape[1:])

            img = self.transform(stacked_observation, aug_factor)

            # Infer (presence of) leading dimensions: [T,B], [B], or [].
            lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

            conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
            if self.renormalize:
                conv_out = renormalize(conv_out, -3)
            p = self.head(conv_out)

            if self.distributional:
                p = F.softmax(p, dim=-1)
            else:
                p = p.squeeze(-1)

            p = p.view(observation.shape[0],
                       max(1, aug_factor),
                       *p.shape[1:]).mean(1)

            # Restore leading dimensions: [T,B], [B], or [], as input.
            p = restore_leading_dims(p, lead_dim, T, B)

            return p

    def select_action(self, obs):
        # value = self.forward(obs, None, None, None, train=False, eval=True)
        value = self.forward(obs, None, None, train=False, eval=True)

        if self.distributional:
            value = from_categorical(value, logits=False, limit=10)
        return value

    def step(self, state, action):
        next_state, reward_logits = self.dynamics_model(state, action)
        return next_state, reward_logits

    def generate_all_actions(self, batch_size, num_actions, device):
        actions = torch.arange(num_actions).long()
        return actions[:, None].expand(num_actions, batch_size).to(device)

    def generate_real_actions(self, real_actions, batch_size, num_actions, device):
        return real_actions[:, None].expand(batch_size, num_actions).to(device)

    def generate_random_actions(self, num_actions, shape, device):
        return torch.randint(num_actions, shape).long().to(device)

    def reweight(self, mask, num_actions, method='equal', real_weight=None):
        '''
        action_masks: (bs, A), 1 indicating real action
        loss: (bs, A)
        '''
        if method == 'ratio' and real_weight == None:
            print("Invalid Input!")
            raise NameError
        if method == 'equal':
            aug_weight = 1. / num_actions
            real_weight = 1. / num_actions
        elif method == 'ratio':
            aug_weight = (1 - real_weight) / (num_actions - 1)
        else:
            print("Incorrect reweighting method!")
            raise NameError
        
        weights = mask * real_weight + (1 - mask) * aug_weight
        return weights

class MLPHead(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 output_size,
                 hidden_size=-1,
                 pixels=30,
                 noisy=0):
        super().__init__()
        if noisy:
            linear = NoisyLinear
        else:
            linear = nn.Linear
        self.noisy = noisy
        if hidden_size <= 0:
            hidden_size = input_channels*pixels
        self.linears = [linear(input_channels*pixels, hidden_size),
                        linear(hidden_size, output_size)]
        layers = [nn.Flatten(-3, -1),
                  self.linears[0],
                  nn.ReLU(),
                  self.linears[1]]
        self.network = nn.Sequential(*layers)
        if not noisy:
            self.network.apply(weights_init)
        self._output_size = output_size

    def forward(self, input):
        return self.network(input)

    def reset_noise(self):
        for module in self.linears:
            module.reset_noise()

    def set_sampling(self, sampling):
        for module in self.linears:
            module.sampling = sampling


class DQNDistributionalHeadModel(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 output_size,
                 hidden_size=256,
                 pixels=30,
                 n_atoms=51,
                 noisy=0,
                 std_init=0.1):
        super().__init__()
        if noisy:
            linear = NoisyLinear
            self.linears = [linear(input_channels*pixels, hidden_size, std_init=std_init),
                            linear(hidden_size, output_size * n_atoms, std_init=std_init)]
        else:
            linear = nn.Linear
            self.linears = [linear(input_channels*pixels, hidden_size),
                            linear(hidden_size, output_size * n_atoms)]
        layers = [nn.Flatten(-3, -1),
                  self.linears[0],
                  nn.ReLU(),
                  self.linears[1]]
        self.network = nn.Sequential(*layers)
        if not noisy:
            self.network.apply(weights_init)
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input):
        return self.network(input).view(-1, self._output_size, self._n_atoms)

    def reset_noise(self):
        for module in self.linears:
            module.reset_noise()

    def set_sampling(self, sampling):
        for module in self.linears:
            module.sampling = sampling


class DQNDistributionalDuelingHeadModel(torch.nn.Module):
    """An MLP head with optional noisy layers which reshapes output to [B, output_size, n_atoms]."""

    def __init__(self,
                 input_channels,
                 output_size,
                 pixels=30,
                 n_atoms=51,
                 hidden_size=256,
                 grad_scale=2 ** (-1 / 2),
                 noisy=0,
                 std_init=0.1):
        super().__init__()
        if noisy:
            self.linears = [NoisyLinear(pixels * input_channels, hidden_size, std_init=std_init),
                            NoisyLinear(hidden_size, output_size * n_atoms, std_init=std_init),
                            NoisyLinear(pixels * input_channels, hidden_size, std_init=std_init),
                            NoisyLinear(hidden_size, n_atoms, std_init=std_init)
                            ]
        else:
            self.linears = [nn.Linear(pixels * input_channels, hidden_size),
                            nn.Linear(hidden_size, output_size * n_atoms),
                            nn.Linear(pixels * input_channels, hidden_size),
                            nn.Linear(hidden_size, n_atoms)
                            ]
        self.advantage_layers = [nn.Flatten(-3, -1),
                                 self.linears[0],
                                 nn.ReLU(),
                                 self.linears[1]]
        self.value_layers = [nn.Flatten(-3, -1),
                             self.linears[2],
                             nn.ReLU(),
                             self.linears[3]]
        self.advantage_hidden = nn.Sequential(*self.advantage_layers[:3])
        self.advantage_out = self.advantage_layers[3]
        self.advantage_bias = torch.nn.Parameter(torch.zeros(n_atoms), requires_grad=True)
        self.value = nn.Sequential(*self.value_layers)
        self.network = self.advantage_hidden
        self._grad_scale = grad_scale
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input):
        x = scale_grad(input, self._grad_scale)
        advantage = self.advantage(x)
        value = self.value(x).view(-1, 1, self._n_atoms)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def advantage(self, input):
        x = self.advantage_hidden(input)
        x = self.advantage_out(x)
        x = x.view(-1, self._output_size, self._n_atoms)
        return x + self.advantage_bias

    def reset_noise(self):
        for module in self.linears:
            module.reset_noise()

    def set_sampling(self, sampling):
        for module in self.linears:
            module.sampling = sampling


class QL1Head(nn.Module):
    def __init__(self, head, dueling=False, type="noisy advantage"):
        super().__init__()
        self.head = head
        self.noisy = "noisy" in type
        self.dueling = dueling
        self.encoders = nn.ModuleList()
        self.relu = "relu" in type
        value = "value" in type
        advantage = "advantage" in type
        if self.dueling:
            if value:
                self.encoders.append(self.head.value[1])
            if advantage:
                self.encoders.append(self.head.advantage_hidden[1])
        else:
            self.encoders.append(self.head.network[1])

        self.out_features = sum([e.out_features for e in self.encoders])

    def forward(self, x):
        x = x.flatten(-3, -1)
        representations = []
        for encoder in self.encoders:
            encoder.noise_override = self.noisy
            representations.append(encoder(x))
            encoder.noise_override = None
        representation = torch.cat(representations, -1)
        if self.relu:
            representation = F.relu(representation)

        return representation


def weights_init(m):
    if isinstance(m, Conv2dSame):
        torch.nn.init.kaiming_uniform_(m.layer.weight, nonlinearity='linear')
        torch.nn.init.zeros_(m.layer.bias)
    elif isinstance(m, (nn.Conv2d, nn.Linear)):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
        torch.nn.init.zeros_(m.bias)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1, bias=True):
        super(NoisyLinear, self).__init__()
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.sampling = True
        self.noise_override = None
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.bias_sigma = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        if not self.bias:
            self.bias_mu.fill_(0)
            self.bias_sigma.fill_(0)
        else:
            self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
            self.bias_mu.data.uniform_(-mu_range, mu_range)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        # Self.training alone isn't a good-enough check, since we may need to
        # activate .eval() during sampling even when we want to use noise
        # (due to batchnorm, dropout, or similar).
        # The extra "sampling" flag serves to override this behavior and causes
        # noise to be used even when .eval() has been called.
        if self.noise_override is None:
            use_noise = self.training or self.sampling
        else:
            use_noise = self.noise_override
        if use_noise:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


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


class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


class Conv2dModel(torch.nn.Module):
    """2-D Convolutional model component, with option for max-pooling vs
    downsampling for strides > 1.  Requires number of input channels, but
    not input shape.  Uses ``torch.nn.Conv2d``.
    """

    def __init__(
            self,
            in_channels,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            use_maxpool=False,  # if True: convs use stride 1, maxpool downsample.
            head_sizes=None,  # Put an MLP head on top.
            dropout=0.,
            ):
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [in_channels] + channels[:-1]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        conv_layers = [torch.nn.Conv2d(in_channels=ic, out_channels=oc,
            kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
            zip(in_channels, channels, kernel_sizes, strides, paddings)]
        sequence = list()
        for conv_layer, maxp_stride in zip(conv_layers, maxp_strides):
            sequence.extend([conv_layer, nonlinearity()])
            if dropout > 0:
                sequence.append(nn.Dropout(dropout))
            if maxp_stride > 1:
                sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
        self.conv = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
        return self.conv(input)


def init_normalization(channels, type="bn", affine=True, one_d=False):
    assert type in ["bn", "ln", "in", "none", None]
    if type == "bn":
        if one_d:
            return nn.BatchNorm1d(channels, affine=affine)
        else:
            return nn.BatchNorm2d(channels, affine=affine)
    elif type == "ln":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=affine)
        else:
            return nn.GroupNorm(1, channels, affine=affine)
    elif type == "in":
        return nn.GroupNorm(channels, channels, affine=affine)
    elif type == "none" or type is None:
        return nn.Identity()


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_type="bn"):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            init_normalization(out_channels, norm_type),
            Conv2dSame(out_channels, out_channels, 3),
            init_normalization(out_channels, norm_type),
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)
        return out


class Conv2dSame(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias,
                            stride=stride, padding=ka)
        )

    def forward(self, x):
        return self.net(x)


def to_categorical(value, limit=300):
    value = value.float()  # Avoid any fp16 shenanigans
    value = value.clamp(-limit, limit)
    distribution = torch.zeros(value.shape[0], (limit*2+1), device=value.device)
    lower = value.floor().long() + limit
    upper = value.ceil().long() + limit
    upper_weight = value % 1
    lower_weight = 1 - upper_weight
    distribution.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    distribution.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return distribution


def from_categorical(distribution, limit=300, logits=True):
    distribution = distribution.float()  # Avoid any fp16 shenanigans
    if logits:
        distribution = torch.softmax(distribution, -1)
    num_atoms = distribution.shape[-1]
    weights = torch.linspace(-limit, limit, num_atoms, device=distribution.device).float()
    return distribution @ weights


class TransitionModel(nn.Module):
    def __init__(self,
                 channels,
                 num_actions,
                 args=None,
                 blocks=16,
                 hidden_size=256,
                 pixels=36,
                 limit=300,
                 action_dim=6,
                 norm_type="bn",
                 renormalize=True,
                 residual=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.args = args
        self.renormalize = renormalize
        self.residual = residual
        layers = [Conv2dSame(channels+num_actions, hidden_size, 3),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type)]
        for _ in range(blocks):
            layers.append(ResidualBlock(hidden_size,
                                        hidden_size,
                                        norm_type))
        layers.extend([Conv2dSame(hidden_size, channels, 3)])

        self.action_embedding = nn.Embedding(num_actions, pixels*action_dim)

        self.network = nn.Sequential(*layers)
        self.reward_predictor = RewardPredictor(channels,
                                                pixels=pixels,
                                                limit=limit,
                                                norm_type=norm_type)
        self.train()

    def forward(self, x, action):
        batch_range = torch.arange(action.shape[0], device=action.device)
        action_onehot = torch.zeros(action.shape[0],
                                    self.num_actions,
                                    x.shape[-2],
                                    x.shape[-1],
                                    device=action.device)
        action_onehot[batch_range, action, :, :] = 1
        stacked_image = torch.cat([x, action_onehot], 1)
        next_state = self.network(stacked_image)
        if self.residual:
            next_state = next_state + x
        next_state = F.relu(next_state)
        if self.renormalize:
            next_state = renormalize(next_state, 1)
        next_reward = self.reward_predictor(next_state)
        return next_state, next_reward


class RewardPredictor(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_size=1,
                 pixels=36,
                 limit=300,
                 norm_type="bn"):
        super().__init__()
        self.hidden_size = hidden_size
        layers = [nn.Conv2d(input_channels, hidden_size, kernel_size=1, stride=1),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type),
                  nn.Flatten(-3, -1),
                  nn.Linear(pixels*hidden_size, 256),
                  nn.ReLU(),
                  nn.Linear(256, limit*2 + 1)]
        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x):
        return self.network(x)


def renormalize(tensor, first_dim=1):
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min)/(max - min)

    return flat_tensor.view(*tensor.shape)


class InverseTransitionModel(nn.Module):
    def __init__(self,
                 channels,
                 num_actions,
                 args=None,
                 blocks=16,
                 hidden_size=256,
                 pixels=36,
                 limit=300,
                 action_dim=6,
                 norm_type="bn",
                 renormalize=True,
                 residual=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.args = args
        self.renormalize = renormalize
        self.residual = residual
        layers = [Conv2dSame(channels*2, hidden_size, 3),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type)]
        for _ in range(blocks):
            layers.append(ResidualBlock(hidden_size,
                                        hidden_size,
                                        norm_type))
        layers.extend([Conv2dSame(hidden_size, channels, 3)])
        self.action_embedding = nn.Embedding(num_actions, pixels*action_dim)

        self.network = nn.Sequential(*layers)
        self.max_pool = nn.AdaptiveMaxPool2d((1,1))
        self.cls = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, num_actions)
        )

        self.train()

    def forward(self, state, next_state):
        ''' s_t+1, s_t '''
        stacked_states = torch.cat([state, next_state], 1)
        feat = self.network(stacked_states)
        action_logits = self.cls(self.max_pool(feat)[:,:,0,0])
        return action_logits


class BackwardTransitionModel(nn.Module):
    def __init__(self,
                 channels,
                 num_actions,
                 args=None,
                 blocks=16,
                 hidden_size=256,
                 pixels=36,
                 limit=300,
                 action_dim=6,
                 norm_type="bn",
                 renormalize=True,
                 residual=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.args = args
        self.renormalize = renormalize
        self.residual = residual
        layers = [Conv2dSame(channels+num_actions, hidden_size, 3),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type)]
        for _ in range(blocks):
            layers.append(ResidualBlock(hidden_size,
                                        hidden_size,
                                        norm_type))
        layers.extend([Conv2dSame(hidden_size, channels, 3)])

        self.action_embedding = nn.Embedding(num_actions, pixels*action_dim)

        self.network = nn.Sequential(*layers)
        # self.reward_predictor = RewardPredictor(channels,
        #                                         pixels=pixels,
        #                                         limit=limit,
        #                                         norm_type=norm_type)
        self.train()

    def forward(self, x, action):
        ''' s_t+1, a_t => s_t '''
        batch_range = torch.arange(action.shape[0], device=action.device)
        action_onehot = torch.zeros(action.shape[0],
                                    self.num_actions,
                                    x.shape[-2],
                                    x.shape[-1],
                                    device=action.device)
        action_onehot[batch_range, action, :, :] = 1
        stacked_image = torch.cat([x, action_onehot], 1)
        next_state = self.network(stacked_image)
        if self.residual:
            next_state = next_state + x
        next_state = F.relu(next_state)
        if self.renormalize:
            next_state = renormalize(next_state, 1)
        # next_reward = self.reward_predictor(next_state)
        # return next_state, next_reward
        return next_state
