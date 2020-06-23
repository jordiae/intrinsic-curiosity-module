import torch
import argparse
import logging
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from rl_curiosity.utils import load_agent, load_icm, LazyTransition, EagerTransition, ReplayBuffer, transform
import os
import gym
from gym import wrappers
import math
from typing import List, Optional, Tuple
import torch.nn.functional as F
import dataclasses
from rl_curiosity.utils import evaluate
from pprint import pformat

# Some parts are inspired by https://github.com/dxyang/DQN_pytorch/blob/master/learn.py


def optimize(transitions: List[EagerTransition], current_model: nn.Module, target_model: nn.Module,
             optimizer, device: torch.device, gamma: float, loss_function: str, curiosity: Optional[nn.Module] = None,
             curiosity_optimizer: Optional = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    transitions = [dataclasses.astuple(t) for t in transitions]
    states, actions, new_states, rewards, dones, predicted_next_state_loss, = \
        tuple(map(lambda x: np.array(list(x)), zip(*transitions)))

    current_model.train()

    if not curiosity:
        states, actions, new_states, rewards, dones = (torch.tensor(states).to(device),
                                                       torch.tensor(actions).long().to(device),
                                                       torch.tensor(new_states).to(device),
                                                       torch.tensor(rewards).to(device),
                                                       torch.tensor(dones).long().to(device)
                                                       )
    else:
        # Instead of r = ri + re
        # r = ri
        states, actions, new_states, rewards, dones = (torch.tensor(states).to(device),
                                                       torch.tensor(actions).long().to(device),
                                                       torch.tensor(new_states).to(device),
                                                       #  torch.tensor(rewards).to(device) +  # no extrinsic
                                                       torch.tensor(predicted_next_state_loss).to(device),
                                                       torch.tensor(dones).long().to(device)
                                                       )

    # Optimize agent
    optimizer.zero_grad()

    current_q_values = current_model(states)
    next_q_values = current_model(new_states)
    next_q_state_values = target_model(new_states)

    current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_values = rewards + gamma * next_q_values * (1 - dones)

    if loss_function == 'mse':
        loss = ((current_q_values - expected_q_values)**2).mean()
    elif loss_function == 'huber':
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)
    else:
        raise NotImplementedError(loss_function)

    loss.backward()
    optimizer.step()

    # Optimize curiosity
    if curiosity:
        curiosity.train()
        curiosity_optimizer.zero_grad()
        state_loss, action_loss = curiosity(states, new_states, actions)
        icm_loss = state_loss + action_loss
        icm_loss.backward()
        curiosity_optimizer.step()

    return loss.data, icm_loss.data if curiosity else None


def train(args: argparse.Namespace, env: gym.Env, exp_dir: str):
    seed = args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    device = torch.device('cuda:0' if not args.no_cuda and torch.cuda.is_available() else 'cpu')

    env = wrappers.FrameStack(wrappers.AtariPreprocessing(env), num_stack=args.stacked_frames)

    writer = SummaryWriter(log_dir=exp_dir)
    with open(os.path.join(exp_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    logging.info(args)

    n_actions = env.action_space.n

    current_model = load_agent(args, env).to(device)
    current_model.eval()
    target_model = load_agent(args, env).to(device)
    target_model.eval()

    if args.curiosity:
        curiosity = load_icm(args, env).to(device)
        curiosity.eval()

    target_model.load_state_dict(current_model.state_dict())  # Sync/update target model

    # rms-prop? https://www.reddit.com/r/reinforcementlearning/comments/ei9p3y/using_rmsprop_over_adam/
    if args.optimizer == 'adam':
        optimizer = optim.Adam(current_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.curiosity:
            curiosity_optimizer = optim.Adam(curiosity.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        logging.error('Optimizer not implemented')
        raise NotImplementedError()

    logging.info(current_model)
    if args.curiosity:
        logging.info(curiosity)
    n_params = sum(p.numel() for p in current_model.parameters() if p.requires_grad)
    logging.info(f'Training {n_params} parameters')
    if args.curiosity:
        n_params = sum(p.numel() for p in curiosity.parameters() if p.requires_grad)
        logging.info(f'Training {n_params} parameters')

    criterion = nn.SmoothL1Loss if args.criterion == 'huber' else None
    if criterion is None:
        raise NotImplementedError(args.criterion)

    buffer = ReplayBuffer(capacity=args.replay_size, seed=args.seed)

    best_mean_reward = env.reward_range[0]
    episodes_without_improvement = 0

    # Adapted from Mario Martin's Notebook
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 10000
    epsilon_by_episode = lambda e: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * e / epsilon_decay)

    if args.curiosity:
        epsilon_by_episode = lambda e: 0.0  # No gamma needed if curiosity is used

    frame_idx = 0

    t0 = time.time()
    all_rewards = []
    all_steps = []
    all_mean_rewards = []
    all_mean_steps = []

    episode_set_rewards = 0.0
    episode_set_curiosity_rewards = 0.0
    episode_set_steps = 0

    updates = 0
    optimizations = 0

    for episode in range(args.episodes):
        state = env.reset()
        frame_idx += 1
        episode_reward = 0.0
        episode_curiosity_reward = 0.0
        steps = 0
        gamma = epsilon_by_episode(episode)
        while True:
            current_model.eval()
            if args.curiosity:
                curiosity.eval()
            action = current_model.act(torch.tensor(transform(state.__array__())).unsqueeze(0).to(device),
                                       gamma,
                                       torch.rand(1)[0].to(device),
                                       torch.randint(0, n_actions, (1,))[0].to(device))
            current_model.train()

            next_state, reward, done, _ = env.step(action)

            episode_reward += reward

            curiosity_reward = None
            if args.curiosity:
                with torch.no_grad():
                    curiosity_reward, _ = \
                        curiosity(torch.tensor(transform(state.__array__())).unsqueeze(0),
                                  torch.tensor(transform(next_state.__array__())).unsqueeze(0).to(device),
                                  torch.tensor([action]).long().to(device))
                episode_curiosity_reward += curiosity_reward

            frame_idx += 1
            buffer.push(LazyTransition(state, action, next_state, reward, done,
                                       curiosity_reward.numpy() if curiosity_reward is not None else None))

            if done:
                writer.add_scalar('Reward/train', episode_reward, episode + 1)
                writer.add_scalar('Steps/train', steps, episode + 1)
                writer.add_scalar('Gamma/train', gamma, episode + 1)
                all_rewards.append(episode_reward)
                all_steps.append(steps)
                episode_set_rewards += episode_reward
                episode_set_steps += steps
                if args.curiosity:
                    writer.add_scalar('Curiosity/train', episode_curiosity_reward, episode + 1)
                    episode_set_curiosity_rewards += episode_curiosity_reward

            state = next_state
            steps += 1

            if args.render:
                env.render()

            if done:
                logging.info(f'Finished episode {episode+1} with reward = {episode_reward:.2f} | steps = {steps+1} | '
                             f'gamma = {gamma:.2f}')
                if args.curiosity:
                    logging.info(f'curiosity = {curiosity_reward:.2f}')
                break

        if buffer.full and (episode+1) % args.optimize_freq == 0:  # len(buffer) >= args.batch_size:

            transitions = buffer.sample(args.batch_size)

            if not args.curiosity:
                q_loss, _ = optimize(transitions, current_model, target_model, optimizer, device, args.gamma,
                                     args.criterion)
            else:
                q_loss, curiosity_loss = optimize(transitions, current_model, target_model, optimizer, device,
                                                  args.gamma, args.criterion, curiosity, curiosity_optimizer)

            mean_episode_set_rewards = episode_set_rewards / (args.optimize_freq-1)
            mean_episode_set_steps = episode_set_steps / (args.optimize_freq-1)
            writer.add_scalar('Mean-Reward/train', mean_episode_set_rewards, optimizations + 1)
            writer.add_scalar('Mean-Steps/train', mean_episode_set_steps, optimizations + 1)
            writer.add_scalar('Q-Loss/train', q_loss, optimizations + 1)
            if args.curiosity:
                writer.add_scalar('Curiosity-Loss/train', curiosity_loss, optimizations + 1)
            all_mean_rewards.append(mean_episode_set_rewards)
            all_mean_steps.append(mean_episode_set_steps)
            episode_set_rewards = 0.0
            episode_set_steps = 0

            torch.save(current_model.state_dict(), os.path.join(exp_dir, 'checkpoint_last.pt'))
            if args.curiosity:
                torch.save(curiosity.state_dict(), os.path.join(exp_dir, 'curiosity_checkpoint_last.pt'))

            logging.info(f'Optimized model ({optimizations+1} optimizations)')
            optimizations += 1

            if mean_episode_set_rewards > best_mean_reward:
                episodes_without_improvement = 0
                best_mean_reward = mean_episode_set_rewards
                torch.save(current_model.state_dict(), os.path.join(exp_dir, 'checkpoint_best.pt'))
                logging.info(f'NEW: Best mean reward: {best_mean_reward:.2f}')
                if best_mean_reward == env.reward_range[1]:
                    logging.info('Reached max reward')
                    break
            else:
                episodes_without_improvement += 1
                logging.info(f'Best mean reward: {best_mean_reward:.2f}')
                if args.early_stop != -1 and episodes_without_improvement == args.early_stop:
                    break
            logging.info(f'{episodes_without_improvement} episodes without improvement')
        elif not buffer.full and episode % args.optimize_freq == 0:
            # First iteration: buffer not full. This messes up the logs (no other effect).
            episode_set_rewards = 0.0
            episode_set_steps = 0.0

        if buffer.full and (episode+1) % args.update_target_freq == 0:
            target_model.load_state_dict(current_model.state_dict())
            logging.info(f'Updated target model (updates {updates})')
            updates += 1

    t1 = time.time()
    logging.info(f'Finished training in {t1-t0:.1f}s')
    if args.render:
        env.close()
    model = load_agent(args, env).to(device)
    model.load_state_dict(torch.load('checkpoint_best.pt'))
    eval_res = evaluate(model, env, args, device, episodes=args.optimize_freq)

    logging.info(pformat(eval_res))
