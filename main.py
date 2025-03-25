import os
print("imported os")
import gymnasium as gym
print("imported gym")
import random
print("imported random")
import argparse
print("imported argparse")
import numpy as np
print("imported numpy")
import tensorflow as tf
print("imported tensorflow")
from networks import CartPoleNetwork
print("imported networks")
from self_play import self_play
print("imported self_play")
from replay import ReplayBuffer
print("imported ReplayBuffer")
from config import get_cartpole_config
print("Imports Finished")

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


SEED = 0


def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


if __name__ == "__main__":
    print("started")
    parser = argparse.ArgumentParser(description='Train MuZero.')
    parser.add_argument('--num_simulations', type=int, default=50)
    args = parser.parse_args()
    print("num simulations:", args.num_simulations)

    # Set seeds for reproducibility
    set_seeds()
    print("seeds set")
    # Create the cartpole network
    network = CartPoleNetwork(
        action_size=2, state_shape=(None, 4), embedding_size=4, max_value=200)
    print("Network Made")
    # Set the configuration for muzero
    config = get_cartpole_config(args.num_simulations)  # Create Environment
    env = gym.make('CartPole-v0')
    print("gym set")

    # Create buffer to store games
    replay_buffer = ReplayBuffer(config)
    self_play(env, config, replay_buffer, network)
    print("played")
