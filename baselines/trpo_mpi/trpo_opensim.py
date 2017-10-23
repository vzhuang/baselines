import opensim as osim
import numpy as np
import sys
import tensorflow as tf

from osim.env import *
from osim.http.client import Client

import argparse
import math
import scipy.io
import time

from mpi4py import MPI
import os.path as osp

from baselines import logger
from baselines.common import set_global_seeds
import logging
from baselines import logger
from baselines.common.mpi_fork import mpi_fork
from baselines import bench
from baselines.trpo_mpi import trpo_mpi
from baselines.pposgd.mlp_policy import MlpPolicy
import sys
num_cpu=1

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', default=10000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--token', dest='token', required=False)
parser.add_argument('--restore', dest='restore', action='store_true', default=False)
args = parser.parse_args()

# Load walking environment
env = RunEnv(args.visualize)
env.reset()

nb_actions = env.action_space.shape[0]

################## BASELINES CODE ##############################################

st = time.time()

seed = 12345
num_timesteps = args.steps
# set_global_seeds(seed)
env.seed(seed)

import baselines.common.tf_util as U
sess = U.single_threaded_session()
sess.__enter__()
# if args.restore:
#     saver = tf.train.Saver()
#     saver.restore(sess, 'saved/trpo_checkpoint')

rank = MPI.COMM_WORLD.Get_rank()
if rank != 0:
    logger.set_level(logger.DISABLED)
workerseed = seed + 1000 * MPI.COMM_WORLD.Get_rank()
set_global_seeds(workerseed)
def policy_fn(name, ob_space, ac_space):
    return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
        hid_size=128, num_hid_layers=2)

env.seed(workerseed)
gym.logger.setLevel(logging.WARN)

trpo_mpi.learn(sess, args.restore, env, policy_fn, timesteps_per_batch=2048, max_kl=0.5, cg_iters=20, cg_damping=0.1,
    max_timesteps=num_timesteps, gamma=0.995, lam=0.97, vf_iters=5, vf_stepsize=5e-3)
# saver = tf.train.Saver()
# saver.save(sess, 'trpo_checkpoint')

print(time.time()-st)

env.close()
