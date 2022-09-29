# --------------------------------------------------------
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import copy
import json
import math
import os
# os.environ['MUJOCO_GL'] = 'egl'
import random
import sys
import time

import dmc2gym
import gym
import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torchvision import transforms

import utils
from curl_sac import CurlSacAgent
from logger import Logger
from spr_sac import SPRSacAgent
from cycdm_sac import CycDMSacAgent
from mtm_sac import MTMSacAgent
from idm_sac import IDMSacAgent
from base_sac import BaseSacAgent
from video import VideoRecorder


def evaluate(env, agent, video, num_episodes, L, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()

            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            while not done:
                # center crop image
                if args.encoder_type == 'pixel':
                    obs = utils.center_crop_image(obs, args.image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward

            video.save('%d.mp4' % step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)

        L.log('eval/' + prefix + 'eval_time', time.time() - start_time, step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'base_sac':
        return BaseSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            augmentation=args.augmentation,
            transition_model_type=args.transition_model_type,
            transition_model_layer_width=args.transition_model_layer_width,
            jumps=args.jumps,
            latent_dim=args.latent_dim,
            num_aug_actions=args.num_aug_actions,
            loss_space=args.loss_space,
            bp_mode=args.bp_mode,
            cycle_steps=args.cycle_steps,
            cycle_mode=args.cycle_mode,
            fp_loss_weight=args.fp_loss_weight,
            bp_loss_weight=args.bp_loss_weight,
            rc_loss_weight=args.rc_loss_weight,
            vc_loss_weight=args.vc_loss_weight,
            reward_loss_weight=args.reward_loss_weight,
            time_offset=args.time_offset,
            momentum_tau=args.momentum_tau,
            aug_prob=args.aug_prob,
            auxiliary_task_lr=args.auxiliary_task_lr,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim,
            sigma=args.sigma)
    if args.agent == 'mtm_sac':
        return MTMSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            augmentation=args.augmentation,
            transition_model_type=args.transition_model_type,
            transition_model_layer_width=args.transition_model_layer_width,
            jumps=args.jumps,
            latent_dim=args.latent_dim,
            num_aug_actions=args.num_aug_actions,
            loss_space=args.loss_space,
            bp_mode=args.bp_mode,
            cycle_steps=args.cycle_steps,
            cycle_mode=args.cycle_mode,
            fp_loss_weight=args.fp_loss_weight,
            bp_loss_weight=args.bp_loss_weight,
            rc_loss_weight=args.rc_loss_weight,
            vc_loss_weight=args.vc_loss_weight,
            reward_loss_weight=args.reward_loss_weight,
            time_offset=args.time_offset,
            momentum_tau=args.momentum_tau,
            aug_prob=args.aug_prob,
            auxiliary_task_lr=args.auxiliary_task_lr,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim,
            sigma=args.sigma,
            mask_ratio=args.mask_ratio,
            patch_size=args.patch_size,
            block_size=args.block_size,
            num_attn_layers=args.num_attn_layers)
    elif args.agent == 'curl_sac':
        return CurlSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim)
    elif args.agent == 'cycdm_sac':
        return CycDMSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            augmentation=args.augmentation,
            transition_model_type=args.transition_model_type,
            transition_model_layer_width=args.transition_model_layer_width,
            jumps=args.jumps,
            latent_dim=args.latent_dim,
            num_aug_actions=args.num_aug_actions,
            loss_space=args.loss_space,
            bp_mode=args.bp_mode,
            cycle_steps=args.cycle_steps,
            cycle_mode=args.cycle_mode,
            fp_loss_weight=args.fp_loss_weight,
            bp_loss_weight=args.bp_loss_weight,
            rc_loss_weight=args.rc_loss_weight,
            vc_loss_weight=args.vc_loss_weight,
            reward_loss_weight=args.reward_loss_weight,
            time_offset=args.time_offset,
            momentum_tau=args.momentum_tau,
            aug_prob=args.aug_prob,
            auxiliary_task_lr=args.auxiliary_task_lr,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim,
            sigma=args.sigma)
    elif args.agent == 'idm_sac':
        return IDMSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            augmentation=args.augmentation,
            transition_model_type=args.transition_model_type,
            transition_model_layer_width=args.transition_model_layer_width,
            jumps=args.jumps,
            latent_dim=args.latent_dim,
            num_aug_actions=args.num_aug_actions,
            loss_space=args.loss_space,
            bp_mode=args.bp_mode,
            cycle_steps=args.cycle_steps,
            cycle_mode=args.cycle_mode,
            fp_loss_weight=args.fp_loss_weight,
            bp_loss_weight=args.bp_loss_weight,
            rc_loss_weight=args.rc_loss_weight,
            vc_loss_weight=args.vc_loss_weight,
            reward_loss_weight=args.reward_loss_weight,
            time_offset=args.time_offset,
            momentum_tau=args.momentum_tau,
            aug_prob=args.aug_prob,
            auxiliary_task_lr=args.auxiliary_task_lr,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim,
            sigma=args.sigma)
    elif args.agent == 'spr_sac':
        return SPRSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            augmentation=args.augmentation,
            transition_model_type=args.transition_model_type,
            transition_model_layer_width=args.transition_model_layer_width,
            jumps=args.jumps,
            latent_dim=args.latent_dim,
            num_aug_actions=args.num_aug_actions,
            loss_space=args.loss_space,
            bp_mode=args.bp_mode,
            cycle_steps=args.cycle_steps,
            cycle_mode=args.cycle_mode,
            fp_loss_weight=args.fp_loss_weight,
            bp_loss_weight=args.bp_loss_weight,
            rc_loss_weight=args.rc_loss_weight,
            vc_loss_weight=args.vc_loss_weight,
            reward_loss_weight=args.reward_loss_weight,
            time_offset=args.time_offset,
            momentum_tau=args.momentum_tau,
            aug_prob=args.aug_prob,
            auxiliary_task_lr=args.auxiliary_task_lr,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim,
            sigma=args.sigma)
    else:
        assert 'agent is not supported: %s' % args.agent


@hydra.main(config_name="config")
def main(args: DictConfig) -> None:
    args.init_steps *= args.action_repeat
    args.log_interval *= args.action_repeat
    args.actor_update_freq *= args.action_repeat
    args.critic_target_update_freq *= args.action_repeat
    args.seed = args.seed or args.seed_and_gpuid[0]
    args.gpuid = args.gpuid or args.seed_and_gpuid[1]
    args.domain_name = args.domain_name or args.env_name.split('/')[0]
    args.task_name = args.task_name or args.env_name.split('/')[1]
    if args.seed == -1:
        args.seed = np.random.randint(1, 1000000)
    torch.cuda.set_device(args.gpuid)
    utils.set_seed_everywhere(args.seed)
    env = dmc2gym.make(domain_name=args.domain_name,
                       task_name=args.task_name,
                       seed=args.seed,
                       visualize_reward=False,
                       from_pixels=args.encoder_type == 'pixel',
                       height=args.pre_transform_image_size,
                       width=args.pre_transform_image_size,
                       frame_skip=args.action_repeat)

    env.seed(args.seed)

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m%dT%H", ts)
    ori_exp_name = 'Ablation-FeatMasking'    # time span - mask ratio - block size - grid size
    # ori_exp_name = 'NoAction-K40'    # re-run
    # ori_exp_name = 'Enc-L6-K32-BLK8-Mask50'    # re-run
    # ori_exp_name = 'MR050_PS7_Mask_Dec32_K6W100_ABSZ256'    # re-run
    # ori_exp_name = 'Baseline-BSZ512'    # re-run
    env_name = ori_exp_name + '-' + args.domain_name + '-' + args.task_name
    exp_name = env_name + ts + '-im' + str(args.image_size) +'-b'  \
    + str(args.batch_size) + '-s' + str(args.seed) + '-' + args.agent

    if args.wandb:
        # wandb.init(project=f'{args.agent}_dmc',
        wandb.init(project=f'MTM-DMC-v2-final',
        # wandb.init(project=f'MTM-DMC-Ablation',
        # wandb.init(project=f'AblateMask-Cartpole',
        # wandb.init(project=f'AblateMask-Cheetah',
        # wandb.init(project=f'AblateMask-Reacher',
        # wandb.init(project=f'AblateMask-Walker',
        # wandb.init(project=f'AblateMask-Ball',
        # wandb.init(project=f'AblateMask-Finger',
                   config=args,
                   name=exp_name,
                #    group=ts,
                   group=ori_exp_name,
                   tags=[
                       env_name,
                       args.agent,
                       str(args.seed),
                       ts.split('T')[0],
                   ],
                   settings=wandb.Settings(start_method="fork"))

    args.work_dir = args.work_dir + '/' + f'{args.agent}_dmc' + '/' + exp_name
    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3 * args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3 * args.frame_stack,
                             args.pre_transform_image_size,
                             args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
        auxiliary_task_batch_size=args.auxiliary_task_batch_size,
        jumps=args.jumps,
    )

    agent = make_agent(obs_shape=obs_shape,
                       action_shape=action_shape,
                       args=args,
                       device=device)
    replay_buffer.add_agent(agent)

    L = Logger(args.work_dir, use_tb=args.save_tb, use_wandb=args.wandb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    for step in range(0, args.num_env_steps, args.action_repeat):
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            evaluate(env, agent, video, args.num_eval_episodes, L, step, args)
            if args.save_model:
                try:
                    if args.agent == 'spr_sac':
                        agent.save_spr(model_dir, step)
                    elif args.agent == 'curl_sac':
                        agent.save_curl(model_dir, step)
                    else:
                        agent.save_cycdm(model_dir, step)
                except:
                    pass
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)    # BGR not RGB

        # import cv2
        # left_im = next_obs[:3].transpose(1, 2, 0)
        # left_im = (((left_im - left_im.min()) / (left_im.max() - left_im.min()) * 1.) * 255).astype(np.uint8)
        # im = left_im[:, :, ::-1]
        # cv2.imwrite('raw3.png', im)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done)
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
