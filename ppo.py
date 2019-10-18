import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import time

from mpeg_creator import MPEGCreator

"""
To check

How many times does PPO use experience? is it 4 batch for 1/4th size?
"""

# number of frame stacks to use
n_stacks = 4

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

def preprocess(raw_obs):
    """ Converts atari frame into cropped 84x84 grayscale image. """
    raw_obs = np.mean(raw_obs, axis=2, keepdims=True) # convert to grayscale.
    raw_obs = cv2.resize(raw_obs, (84, 84))
    raw_obs = raw_obs[:, :, np.newaxis]
    return raw_obs


def write_cv2_video(filename, frames):

    height, width, channels = frames[0].shape

    #print("writing video "+filename+" "+str(frames[0].shape)+" "+str(len(frames)))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height), isColor=True)

    for frame in frames:
        out.write(frame)

    out.release()

class FrameStack():

    def __init__(self, n_stacks):
        self.frames = [np.zeros((84, 84, 1)) for _ in range(n_stacks)]
        self.n_stacks = n_stacks

    def push(self, frame):
        self.frames = self.frames[1:] + [frame]

    def get(self):
        return np.concatenate(self.frames, axis=2)

    def peek(self):
        return self.frames[-1]

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
        x = torch.from_numpy(x).float()/255
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

def entropy(p):
   return (-p * p.log2()).sum()

def log_entropy(logp):
    p = logp.exp()
    return (-p * p.log2()).sum()

def smooth(X, alpha=0.95):
    y = X[0]
    results = []
    for x in X:
        y = (1-alpha) * x + (alpha) * y
        results.append(y)
    return results


def export_video(filename, frames, scale=4):

    if len(frames) == 0:
        return

    height, width, channels = frames[0].shape

    blank_frame = np.zeros((int(height*scale), int(width*scale), 3), dtype=np.uint8) * 0 + 128

    processed_frames = []

    for frame in frames:

        # convert single channel grayscale to rgb grayscale

        if channels == 1:
            frame = np.concatenate([frame] * 3, axis=2)

        # frame marking
        frame[1, 1, 0] = 0
        frame[1, 1, 1] = np.random.randint(0, 255)
        frame[1, 1, 2] = np.random.randint(0, 255)

        if scale != 1:
            frame = cv2.resize(frame, (height * scale, width * scale), interpolation=cv2.INTER_NEAREST)

        frame = np.uint8(frame)

        processed_frames.append(frame)

    write_cv2_video(filename,processed_frames)

def train(env_name, model: nn.Module):

    trace("Training started.")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    N = 100             # steps per update
    gamma = 0.99        # discount
    lamb = 0.95         # GAE parameter
    batch_size = 32
    epsilon = 0.2
    VF_coef = 1.0       # how much loss to take from value function
    agents = 1          # stub
    epochs = 4          # stub

    envs = [gym.make(env_name) for _ in range(agents)]

    # initialize agent
    states = [FrameStack(n_stacks) for _ in envs]

    for state, env in zip(states, envs):
        obs = preprocess(env.reset())
        for _ in range(n_stacks):
            state.push(obs)

    episode_score = [0 for _ in envs]

    training_log = []

    score_history = [0]

    video_frames = []

    for step in range(100000):

        # collect experience

        # history is
        # previous observation
        # action a at t
        # observation
        # reward for taking action a at time t
        # value estimation *before* taking action
        # probabilities of taking these actions.

        agent_history = []
        agent_advanage = []

        for i, env in enumerate(envs):

            state = states[i]
            history = []

            for t in range(N):

                if i == 0:
                    video_frames.append(state.peek())

                stack = state.get()

                probs = torch.exp(model.forward(stack).detach())
                action = sample_action(probs)
                value_estimate = model.value(stack).detach()

                observation, reward, done, info = env.step(action)

                prev_stack = stack.copy()

                state.push(preprocess(observation))
                stack = state.get()

                history.append((prev_stack, action, stack, reward, value_estimate, probs))
                episode_score[i] += reward

                if done:
                    observation = preprocess(env.reset())
                    for _ in range(n_stacks):
                        state.push(observation)
                    score_history.append(episode_score[i])
                    episode_score[i] = 0
                    if i == 0:
                        export_video("video {}.mp4".format(step), video_frames)
                        video_frames = []

            states[i] = state

            # generate advantage estimates
            # note, this can be done much faster, but this will do for the moment.

            advantages = []
            for t in range(N):
                advantage_t = 0
                for i in range(t, N-1):
                    _, _, _, reward_i, value_i, _ = history[i]
                    _, _, _, _, value_next_i, _ = history[i+1]
                    delta_i = reward_i + gamma * value_next_i - value_i
                    advantage_t += ((lamb*gamma)**i)*delta_i
                advantages.append(advantage_t)

            agent_history.extend(history)
            agent_advanage.extend(advantages)

        agent_history = np.array(agent_history)
        agent_advanage = np.array(agent_advanage)

        # export video preview of agent.
        export_video("preview.mp4", video_frames)

        # train on this experience using PPO
        for i in range(epochs):

            ordering = list(range(len(agent_history)))
            np.random.shuffle(ordering)
            agent_history = np.array([agent_history[i] for i in ordering])
            agent_advantage = np.array([agent_advanage[i] for i in ordering])

            batches = len(agent_history) // batch_size

            reward_sum = 0

            total_loss = 0
            total_loss_clip = 0
            total_loss_value = 0
            total_loss_entropy = 0
            total_reward_sum = 0

            for j in range(batches):

                optimizer.zero_grad()
                # compute losses
                loss_clip = torch.zeros([1])
                loss_value = torch.zeros([1])
                loss_entropy = torch.zeros([1])

                batch_sample = list(range(j*batch_size,(j+1)*batch_size))

                for sample in batch_sample:
                    prev_observation, action, observation, reward, value_estimate, probs = agent_history[sample]
                    advantage = agent_advantage[sample]

                    # to we take the mean here?
                    ratio = (model.forward(prev_observation)[action] / probs[action])
                    loss_clip += (1 / batch_size) * torch.min(ratio * advantage, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage)
                    loss_value -= (1 / batch_size) * VF_coef * (reward + gamma*model.value(observation).detach() - model.value(prev_observation))**2
                    loss_entropy += (1 / batch_size) * 0.01 * entropy(model.forward(prev_observation))
                    reward_sum += reward

                loss = -(loss_clip + loss_value + loss_entropy) # gradient ascent.
                loss.backward()
                optimizer.step()

                total_loss_clip += loss_clip
                total_loss_value += loss_value
                total_loss_entropy += loss_entropy
                total_loss += loss
                total_reward_sum += reward_sum

            training_log.append(
                (float(total_loss),
                 float(total_loss_clip),
                 float(total_loss_value),
                 float(total_loss_entropy),
                 np.mean(score_history[-100:])
                 )
            )

        if step % 10 == 0:
            print("{:<14}{:<14}{:<14}{:<14}{:<14}{:<14}".format("step","loss","loss_clip","loss_value","loss_entropy", "ep_score"))
        print("{:<14}{:<14.3f}{:<14.3f}{:<14.3f}{:<14.3f}{:<14.3f}".format(
            step,
            training_log[-1][0],
            training_log[-1][1],
            training_log[-1][2],
            training_log[-1][3],
            training_log[-1][4]
        ))

        if step % 10 == 0:
            xs = range(len(training_log))
            plt.figure(figsize=(8, 8))
            plt.plot(xs, smooth([x[0] for x in training_log]), label='loss')
            plt.plot(xs, smooth([x[1] for x in training_log]), label='loss_clip')
            plt.plot(xs, smooth([x[2] for x in training_log]), label='loss_value')
            plt.plot(xs, smooth([x[3] for x in training_log]), label='loss_entropy')
            plt.legend()
            plt.savefig('loss.png')

            plt.figure(figsize=(8, 8))
            plt.plot(xs, smooth([x[4] for x in training_log]), label='reward')
            plt.legend()
            plt.savefig('reward.png')

    return training_log


def main():
    env_name = "Pong-v4"

    n_actions = gym.make(env_name).action_space.n

    print("Playing {} with {} actions.".format(env_name, n_actions))

    model = MLPModel((84,84,n_stacks), n_actions)
    log = train(env_name, model)
    xs = range(len(log))
    plt.plot(xs, smooth([x for x, y, z, u, v in log]),label='loss')
    plt.plot(xs, smooth([y for x, y, z, u, v in log]),label='loss_clip')
    plt.plot(xs, smooth([z for x, y, z, u, v in log]),label='loss_value')
    plt.legend()
    plt.show()
    plt.plot(xs, smooth([u for x, y, z, u, v in log]), label='reward')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()