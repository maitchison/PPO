import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

"""
To check

How many times does PPO use experience? is it 4 batch for 1/4th size?
"""

def prod(X):
    y = 1
    for x in X:
        y *= x
    return y

def trace(s):
    print(s)

def sample_action(p):
    """ Returns integer [0..len(probs)-1] based on probabilities. """
    p = p.double()
    p /= p.sum() # probs are sometimes off by a little due to precision error
    return np.random.choice(range(len(p)), p=p)

def preprocess(obs):
    """ Converts atari frame into cropped 84x84x3 grayscale image. """
    obs = cv2.resize(obs, (84, 84))
    return obs

class MLPModel(nn.Module):

    """ A very simple Multi Layer Perceptron """
    def __init__(self, input_dims, actions):
        super(MLPModel, self).__init__()
        self.actions = actions
        self.d = prod(input_dims)
        self.fc1 = nn.Linear(self.d, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_policy = nn.Linear(128, actions)
        self.fc_value = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.from_numpy(x).float()
        x = x.view(-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_policy(x)
        x = F.softmax(x, dim=0)
        return x

    def value(self, x):
        x = torch.from_numpy(x).float()
        x = x.view(-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_value(x)
        return x


def train(env: gym.Env, model: nn.Module):

    trace("Training started.")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # initialize agent
    observation = preprocess(env.reset())


    N = 100             # steps per epoch
    gamma = 0.99        # discount
    lamb = 0.95         # GAE parameter
    batch_size = 64
    epsilon = 0.2
    VF_coef = 1.0       # how much loss to take from value function

    for step in range(100):

        trace("Step:" + str(step))

        # collect experience

        # history is
        # previous observation
        # action a at t
        # observation
        # reward for taking action a at time t
        # value estimation *before* taking action
        # probabilities of taking these actions.
        history = []

        for t in range(N):
            prev_observation = observation
            probs = model.forward(observation).detach()
            action = sample_action(probs)
            value_estimate = model.value(observation).detach()
            observation, reward, done, info = env.step(action)
            observation = preprocess(observation)
            history.append((prev_observation, action, observation, reward, value_estimate, probs))

        # generate advantage estimates
        # note, this can be done much faster, but this will do for the moment.

        advantages = []
        for t in range(N):
            advantage_t = 0
            action_t = history[t][1]
            for i in range(t, N-1):
                _, _, _, reward_i, value_i, _ = history[i]
                _, _, _, _, value_next_i, _ = history[i+1]
                delta_i = reward_i + gamma * value_next_i - value_i
                advantage_t += ((lamb*gamma)**(i))*delta_i
            advantages.append(advantage_t)

        # questions:
        # where to gradients go back, just through policy, not policy old right?

        # train on this experience using PPO
        for i in range(4):
            optimizer.zero_grad()
            # compute losses
            loss_clip = torch.zeros([1])
            loss_value = torch.zeros([1])
            for j in range(batch_size):
                sample = np.random.choice(range(len(history)))

                prev_observation, action, observation, reward, value_estimate, probs = history[sample]
                advantage = advantages[sample]

                # to we take the mean here?
                ratio = (model.forward(prev_observation)[action] / probs[action]) * advantage
                loss_clip += (1 / batch_size) * torch.min(ratio * advantage, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage)
                loss_value += VF_coef * (model.value(prev_observation) - (reward + model.value(observation).detach()))**2

            loss = loss_clip + loss_value
            loss.backward()
            optimizer.step()

            print(float(loss), float(loss_clip), float(loss_value))


def main():
    model = MLPModel((84,84,3), 4)
    env = gym.make("Breakout-v4")
    train(env, model)

if __name__ == "__main__":
    main()