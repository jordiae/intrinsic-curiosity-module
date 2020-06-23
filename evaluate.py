import gym
from gym import wrappers
import argparse
import torch
import logging
import time
import os
from rl_curiosity.utils import load_agent, ArgsStruct, evaluate
import json
from pprint import pformat

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate an agent for Atari environments')
    parser.add_argument('path', help='Experiment path', type=str)
    parser.add_argument('--episodes', type=int, help='Number of episodes to be evaluated (set to optimize_freq in train'
                                                     'by default)')
    parser.add_argument('--no-cuda', action='store_true', help='Disables CUDA training')
    parser.add_argument('--env', type=str, help='Environment (if not specified, the same as in training)')
    parser.add_argument('--gamma', type=float, help='Future reward parameter', default=0.999)
    parser.add_argument('--render', action='store_true', help='Render the environment')

    eval_args = parser.parse_args()

    exp_dir = eval_args.path

    with open(os.path.join(eval_args.path, 'args.json'), 'r') as f:
        train_args = ArgsStruct(**json.load(f))

    env_name = eval_args.env if eval_args.env is not None else train_args.env
    episodes = eval_args.episodes if eval_args.episodes is not None else train_args.optimize_freq

    timestamp = time.strftime("%Y-%m-%d-%H%M")
    log_path = os.path.join(exp_dir, f'eval-{timestamp}.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())

    if env_name == 'pong':
        env_id = 'PongNoFrameskip-v0'
        env = gym.make(env_id)
    else:
        raise NotImplementedError(env_name)

    device = torch.device('cuda:0' if not eval_args.no_cuda and torch.cuda.is_available() else 'cpu')

    env = wrappers.FrameStack(wrappers.AtariPreprocessing(env), num_stack=4)

    model = load_agent(train_args, env).to(device)
    model.load_state_dict(torch.load(os.path.join(exp_dir, 'checkpoint_best.pt')))

    eval_res = evaluate(model, env, train_args, device, episodes)

    logging.info(pformat(eval_res))
