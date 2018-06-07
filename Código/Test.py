import argparse
import numpy as np
import gym
import gym_ple
import torch
import numpy as np
from itertools import count
from Agente import Agente 

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical



'''
Policy-related
'''
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


#env = gym.make('CartPole-v0')
env = gym.make('FlappyBird-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

agente = Agente()

def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = agente.select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            agente.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        agente.finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward == 1:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()



