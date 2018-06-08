from Net import Net 
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

class Agente():
    def __init__(self):
        self.policy = Net()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.gamma = 0.8
        self.ant = 0

    def select_action(self, state):
        state = self.preProc(state)
        state = torch.from_numpy(state).float().unsqueeze(0)

        probs = self.policy(state)
        print(probs)
        ac = probs.argmax()
        self.policy.saved_log_probs.append(probs)
        # print (ac)
        return  ac #Manda 0 o 1 dependiendo de cual accion escoge

    def finish_episode(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.policy.rewards[::-1]: #reverse
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        #print(rewards)
        #rewards = (rewards ) / (rewards.std() + self.eps)
        #print(rewards)
        for log_prob, reward in zip(self.policy.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)

        print(policy_loss)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss)
        c = 
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]

    def preProc(self, observation):
        #plt.imshow(observation)
        #plt.pause(0.0000001)
        #plt.imshow(observation[:,:,1], cmap='gray')
        #plt.pause(0.0000001)
        observation = observation[:410,:,1]
        observation[observation > 145] = 255

        if self.ant is None:
            self.ant = observation
        else:
            observation = observation - (self.ant) 

        #plt.imshow(observation, cmap='gray')
        #plt.pause(0.0000001)

        return observation

    def append(self, reward):
        if reward == -5:
            reward = -1
        elif reward == 1:
            reward = 1

        self.policy.rewards.append(reward*5)