from dataclasses import dataclass
import numpy as np
import gym
from typing import List
from gym import wrappers
from torchvision import transforms
from torch import nn
import argparse
import torch
from typing import Dict
import logging
import time
from statistics import stdev


class ArgsStruct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __str__(self) -> str:
        res = 'Namespace('
        for key in self.__dict__:
            res += f'{key}={self.__dict__[key]}, '
        res += ')'
        return res


def load_agent(args: argparse.Namespace, env: gym.Env) -> nn.Module:
    from .model import CNNAgent
    n_actions = env.action_space.n
    input_channels, input_height, input_width = env.observation_space.shape

    if args.arch == 'cnn':
        return CNNAgent(input_height=input_height, input_width=input_width, input_channels=input_channels,
                        dropout=args.dropout, batch_norm=args.batch_norm, n_conv=args.conv_layers, n_fc=args.fc_layers,
                        n_actions=n_actions)

    else:
        raise NotImplementedError(args.model)


def load_vae(args: argparse.Namespace, env: gym.Env) -> nn.Module:
    from .model import VAE
    input_channels, input_height, input_width = env.observation_space.shape

    if args.arch == 'cnn':
        return VAE(input_height=input_height, input_width=input_width, input_channels=input_channels,
                   dropout=args.dropout, batch_norm=args.batch_norm, n_conv=args.conv_layers, z_size=args.z_size)

    else:
        raise NotImplementedError(args.model)


def load_icm(args: argparse.Namespace, env: gym.Env) -> nn.Module:
    from .model import IntrinsicCuriosity
    input_channels, input_height, input_width = env.observation_space.shape

    if args.arch == 'cnn':
        return IntrinsicCuriosity(n_actions=args.n_actions, icm_state_features=args.icm_state_features,
                                  icm_hidden_size=args.icm_hidden_size, icm_n_hidden=args.icm_n_hidden,
                                  input_height=input_height, input_width=input_width,
                                  input_channels=input_channels, dropout=args.dropout, batch_norm=args.batch_norm,
                                  n_conv=args.n_conv)

    else:
        raise NotImplementedError(args.model)


def transform(x: np.ndarray) -> np.ndarray:
    x = x.transpose((1, 2, 0))
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])(x).numpy()


@dataclass
class EagerTransition:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: float
    done: bool


@dataclass
class LazyTransition:
    state: wrappers.LazyFrames
    action: int
    next_state: wrappers.LazyFrames
    reward: float
    done: bool

    def eager(self) -> EagerTransition:
        return EagerTransition(transform(self.state.__array__()), self.action, transform(self.next_state.__array__()),
                               self.reward, self.done)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.memory = [None]*capacity
        self.idx = 0
        self.seed = seed
        np.random.seed(seed)
        self.full = False

    def push(self, transition: LazyTransition):
        self.memory[self.idx] = transition
        if self.idx + 1 == len(self.memory):
            self.full = True
            self.idx = 0
        else:
            self.idx += 1

    def sample(self, batch_size) -> List[EagerTransition]:
        sample = np.random.choice(list(range(len(self.memory))), batch_size)
        return [t.eager() for idx, t in enumerate(self.memory) if idx in sample]

    def __len__(self) -> int:
        return self.idx if not self.full else len(self.memory)


def evaluate(model: nn.Module, env: gym.Env, args: argparse.Namespace, device: torch.device, episodes: int) -> Dict:
    seed = args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    logging.info(args)

    t0 = time.time()
    all_rewards = []
    all_steps = []

    episode_set_rewards = 0.0
    episode_set_steps = 0

    model.eval()

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        steps = 0
        while True:

            action = model.act(torch.tensor(transform(state.__array__())).unsqueeze(0), 0.0,
                               torch.tensor(0.0).to(device), torch.tensor(0).long().to(device))

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            if done:
                all_rewards.append(episode_reward)
                all_steps.append(steps)
                episode_set_rewards += episode_reward
                episode_set_steps += episode_set_steps

            state = next_state
            steps += 1

            if args.render:
                env.render()

            if done:
                logging.info(f'Finished evaluation of episode {episode+1} with reward = {episode_reward+1} and '
                             f'{steps+1} steps')
                break

    t1 = time.time()
    logging.info(f'Finished evaluation in {t1-t0:.1f}s')

    if args.render:
        env.close()

    mean_episode_set_rewards = episode_set_rewards/episodes
    mean_episode_set_steps = episode_set_steps/episodes

    return {'mean_episode_rewards': mean_episode_set_rewards, 'stdev_episode_rewards': stdev(all_rewards),
            'stdev_episode_steps': stdev(all_steps), 'mean_episode_steps': mean_episode_set_steps,
            'episodes': episodes, 'all_rewards': all_rewards,
            'all_steps': all_steps}
