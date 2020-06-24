import argparse
import logging
import time
import os
import gym
from rl_curiosity.trainer import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a double DQN for Atari environments')
    parser.add_argument('experiment_name', help='Experiment name', type=str)
    parser.add_argument('--arch', type=str, help='Architecture', default='cnn')
    parser.add_argument('--episodes', type=int, help='Maximum number of episodes', default=2000000)
    parser.add_argument('--lr', type=float, help='Learning Rate', default=0.0001)
    parser.add_argument('--no-cuda', action='store_true', help='Disables CUDA training')
    parser.add_argument('--optimizer', type=str, help='Optimizer', default='adam')
    parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=32)
    parser.add_argument('--criterion', type=str, help='Criterion (huber or mse)',
                        default='huber')
    parser.add_argument('--early-stop', type=int,
                        help='Episode patience in early stop (-1 -> no early stop)', default=-1)
    parser.add_argument('--weight-decay', type=float, help='Weight decay', default=0.0001)
    parser.add_argument('--dropout', type=float, help='Dropout in FC layers', default=0.25)
    parser.add_argument('--conv-layers', type=int, help='Number of convolutional layers', default=6)
    parser.add_argument('--fc-layers', type=int, help='Number of fully-connected layers', default=2)
    parser.add_argument('--batch-norm', action='store_true', help='Use batch normalization')
    parser.add_argument('--curiosity', action='store_true', help='Use intrinsic reward')
    parser.add_argument('--z-size', type=int, help='Size of the latent vector (only used if --curiosity is set),'
                                                   'sets the size of the state feature vector for the curiosity'
                                                   'modules',
                        default=128)
    parser.add_argument('--stacked-frames', type=int, help='Stacked frames for deal with non-markovian environments'
                                                           'with a CNN-based DQN', default=4)
    parser.add_argument('--env', type=str, help='Environment', default='pong')
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--replay-size', type=int, help='Replay buffer size', default=10000)
    parser.add_argument('--update-target-freq', type=int, help='Episode update frequency of the target model (double'
                                                               'DQN)', default=100)
    parser.add_argument('--optimize-freq', type=int, help='Episode optimization frequency',
                        default=10)
    parser.add_argument('--gamma', type=float, help='Future reward parameter', default=0.999)
    parser.add_argument('--render', action='store_true', help='Render the environment')

    parser.add_argument('--icm-hidden-size', type=int, help='Curiosity module hidden size',
                        default=256)

    parser.add_argument('--icm-n-hidden', type=int, help='Number of hidden layers in curiosity module',
                        default=1)

    args = parser.parse_args()

    timestamp = time.strftime("%Y-%m-%d-%H%M")
    exp_dir = os.path.join('experiments', f'{args.experiment_name}-{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)

    log_path = os.path.join(exp_dir, 'train.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())

    if args.env == 'pong':
        env_id = 'PongNoFrameskip-v0'
        env = gym.make(env_id)
    elif args.env == 'nle':
        raise NotImplementedError(args.env)
        # import nle
        # env = gym.make('NetHackScore-v0')
    else:
        raise NotImplementedError(args.env)

    train(args, env, exp_dir)
