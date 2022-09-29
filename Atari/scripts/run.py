# --------------------------------------------------------
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import copy
from rlpyt.experiments.configs.atari.dqn.atari_dqn import configs
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.atari.atari_env import AtariTrajInfo
from rlpyt.utils.logging.context import logger_context

import os
import wandb
import torch
import numpy as np

from src.models import PVCatDqnModel
from src.rlpyt_utils import OneToOneSerialEvalCollector, SerialSampler, MinibatchRlEvalWandb
from src.algos import PVCategoricalDQN
from src.agent import PVAgent
from src.rlpyt_atari_env import AtariEnv
from src.utils import set_config


def build_and_train(game="pong", run_ID=0, cuda_idx=0, args=None):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = AtariEnv
    # env.seed(1)
    config = set_config(args, game)

    sampler = SerialSampler(
        EnvCls=env,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=config["env"],
        eval_env_kwargs=config["eval_env"],
        batch_T=config['sampler']['batch_T'],
        batch_B=config['sampler']['batch_B'],
        max_decorrelation_steps=0,
        eval_CollectorCls=OneToOneSerialEvalCollector,
        eval_n_envs=config["sampler"]["eval_n_envs"],
        eval_max_steps=config['sampler']['eval_max_steps'],
        eval_max_trajectories=config["sampler"]["eval_max_trajectories"],
    )
    args.discount = config["algo"]["discount"]  # 0.99
    algo = PVCategoricalDQN(optim_kwargs=config["optim"], jumps=args.jumps, **config["algo"])  # Run with defaults.
    agent = PVAgent(ModelCls=PVCatDqnModel, model_kwargs=config["model"], **config["agent"])

    wandb.config.update(config)
    runner = MinibatchRlEvalWandb(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=args.n_steps,
        affinity=dict(cuda_idx=cuda_idx),
        log_interval_steps=args.n_steps//args.num_logs,
        seed=args.seed,
        final_eval_only=args.final_eval_only,
    )
    config = dict(game=game)
    name = "dqn_" + game
    log_dir = "logs"
    print("=======")
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()

    quit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', help='Experiment name', default='Atari')
    parser.add_argument('--game', help='Atari game', default='ms_pacman')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--grayscale', type=int, default=1)
    parser.add_argument('--framestack', type=int, default=4)
    parser.add_argument('--imagesize', type=int, default=84)
    parser.add_argument('--n-steps', type=int, default=100000)
    parser.add_argument('--dqn-hidden-size', type=int, default=256)
    parser.add_argument('--target-update-interval', type=int, default=1)
    parser.add_argument('--target-update-tau', type=float, default=1.)
    parser.add_argument('--momentum-tau', type=float, default=1.)
    parser.add_argument('--batch-b', type=int, default=1)
    parser.add_argument('--batch-t', type=int, default=1)
    parser.add_argument('--beluga', action="store_true")
    parser.add_argument('--jumps', type=int, default=9)
    parser.add_argument('--num-logs', type=int, default=10)
    parser.add_argument('--renormalize', type=int, default=1)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--replay-ratio', type=int, default=64)
    parser.add_argument('--dynamics-blocks', type=int, default=0)
    parser.add_argument('--residual-tm', type=int, default=0.)
    parser.add_argument('--n-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--tag', type=str, default='', help='Tag for wandb run.')
    parser.add_argument('--wandb-dir', type=str, default='./', help='Directory for wandb files.')
    parser.add_argument('--norm-type', type=str, default='bn', choices=["bn", "ln", "in", "none"], help='Normalization')
    parser.add_argument('--aug-prob', type=float, default=1., help='Probability to apply augmentation')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout probability in convnet.')
    parser.add_argument('--spr', type=int, default=1)
    parser.add_argument('--distributional', type=int, default=1)
    parser.add_argument('--delta-clip', type=float, default=1., help="Huber Delta")
    parser.add_argument('--prioritized-replay', type=int, default=1)
    parser.add_argument('--momentum-encoder', type=int, default=1)
    parser.add_argument('--shared-encoder', type=int, default=0)
    parser.add_argument('--local-spr', type=int, default=0)
    parser.add_argument('--global-spr', type=int, default=1)
    parser.add_argument('--noisy-nets', type=int, default=1)
    parser.add_argument('--noisy-nets-std', type=float, default=0.1)
    parser.add_argument('--classifier', type=str, default='q_l1', choices=["mlp", "bilinear", "q_l1", "q_l2", "none"], help='Style of NCE classifier')
    parser.add_argument('--final-classifier', type=str, default='linear', choices=["mlp", "linear", "none"], help='Style of NCE classifier')
    parser.add_argument('--augmentation', type=str, default=["shift", "intensity"], nargs="+",
                        choices=["none", "rrc", "affine", "crop", "blur", "shift", "intensity"],
                        help='Style of augmentation')
    parser.add_argument('--q-l1-type', type=str, default=["value", "advantage"], nargs="+",
                        choices=["noisy", "value", "advantage", "relu"],
                        help='Style of q_l1 projection')
    parser.add_argument('--target-augmentation', type=int, default=1, help='Use augmentation on inputs to target networks')
    parser.add_argument('--eval-augmentation', type=int, default=0, help='Use augmentation on inputs at evaluation time')
    parser.add_argument('--reward-loss-weight', type=float, default=0.)
    parser.add_argument('--model-rl-weight', type=float, default=0.)
    parser.add_argument('--model-spr-weight', type=float, default=5.)
    parser.add_argument('--t0-spr-loss-weight', type=float, default=0.)
    parser.add_argument('--eps-steps', type=int, default=2001)
    parser.add_argument('--min-steps-learn', type=int, default=2000)
    parser.add_argument('--eps-init', type=float, default=1.)
    parser.add_argument('--eps-final', type=float, default=0.)
    parser.add_argument('--final-eval-only', type=int, default=1)
    parser.add_argument('--time-offset', type=int, default=0)
    parser.add_argument('--project', type=str, default="SPR")
    parser.add_argument('--entity', type=str, default="abs-world-models")
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--max-grad-norm', type=float, default=10., help='Max Grad Norm')
    parser.add_argument('--public', action='store_true', help='If set, uses anonymous wandb logging')
    parser.add_argument('--real-cycle', action='store_true')
    parser.add_argument('--virtual-cycle', action='store_true')
    parser.add_argument('--fp', action='store_true', help='if set, then calculate forward prediction loss')
    parser.add_argument('--bp', action='store_true', help='if set, then calculate backward prediction loss')
    parser.add_argument('--bp-mode', type=str, default='gt', help='train RDM starting from gt s_t+K(gt) or estimated s_t+K(esti)')
    parser.add_argument('--rc-weight', type=float, default=0.)
    parser.add_argument('--vc-weight', type=float, default=1.)
    parser.add_argument('--fp-weight', type=float, default=1., help='Forward Prediction Loss weight')
    parser.add_argument('--bp-weight', type=float, default=0., help='Backward Prediction Loss weight')
    parser.add_argument('--mlr-weight', type=float, default=0., help='MLR Loss weight')
    parser.add_argument('--cycle-step', type=int, default=9)
    parser.add_argument('--aug-num', type=float, default=2.)
    parser.add_argument('--space', type=str, default='y', help='calculate similarity loss in cycle on which space: mse, z or y')
    parser.add_argument('--aug-type', type=str, default='random', help='random, nonaug or hybrid')
    parser.add_argument('--warmup', type=int, default=50000)
    parser.add_argument('--bp-warm', action='store_true', help='if set, then annealing decreasing bp-weight')
    args = parser.parse_args()
    print(args)
    
    print("============= Experiment:{} =============".format(args.name))
    os.environ['WANDB_MODE'] = 'dryrun'

    if args.public:
        wandb.init(anonymous="allow", config=args, tags=[args.tag] if args.tag else None, dir=args.wandb_dir)
    else:
        # wandb.init(project=args.project, entity=args.entity, config=args, tags=[args.tag] if args.tag else None, dir=args.wandb_dir)
        wandb.init(project=args.project, config=args, tags=[args.tag] if args.tag else None, dir=args.wandb_dir, name=args.name)
    wandb.config.update(vars(args))
    build_and_train(game=args.game,
                    cuda_idx=args.cuda_idx,
                    args=args)

