import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import time
import itertools
from collections import deque

from mpeg_creator import MPEGCreator

DEVICE = "cuda"

def show_cuda_info():

    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    device_id = torch.cuda.current_device()
    print("Device:", DEVICE)
    print(torch.cuda.get_device_name(device_id))

"""
------------------------------------------------------------------------------------------------------------------------
    Wrappers
------------------------------------------------------------------------------------------------------------------------
"""


class NormalizeObservationWrapper(gym.Wrapper):
    def __init__(self, env, clip=5.0):
        """
        Normalize and clip observations.
        """
        super().__init__(env)
        self.clip = clip
        self._n = 1000
        self.epsilon = 0.00001

        new_space = gym.spaces.Box(
            low = -clip,
            high = clip,
            shape = env.observation_space.shape,
            dtype=np.float32
        )

        self.history = deque(maxlen=self._n)
        self.env = env

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.history.append(obs)

        #note: this will be slow for large observation spaces, would be better to do a running average.
        self.means = np.mean(np.asarray(self.history), axis=0)
        self.stds = np.std(np.asarray(self.history), axis=0)

        obs = np.asarray(obs, dtype=np.float32)
        obs = np.clip((obs - self.means) / (self.stds + self.epsilon), -self.clip, +self.clip)

        return obs, reward, done, info


class NormalizeRewardWrapper(gym.Wrapper):

    def __init__(self, env, clip=5.0):
        """
        Normalizes rewards
        """
        super().__init__(env)

        self.env = env
        self._n = 10000
        self.history = deque(maxlen=self._n)
        self.clip = clip
        self.epsilon = 0.00001

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.history.append(reward)

        self.mean = np.mean(self.history, axis=0)
        self.std = np.std(self.history, axis=0)

        last_raw_reward = reward
        reward = np.clip(reward / (self.std + self.epsilon), -self.clip, +self.clip)
        print(reward, last_raw_reward, self.std, self.history)

        info["raw_reward"] = last_raw_reward

        return obs, reward, done, info


class AtariWrapper(gym.Wrapper):

    def __init__(self, env):
        """
        Stack and do other stuff...
        """
        super().__init__(env)

        self.history = []
        self.env = env

        self.nstacks = 4
        self._width, self._height = 84, 84

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._height, self._width, self.nstacks),
            dtype=np.float32,
        )

    def _push_raw_obs(self, obs):

        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = obs[30:-10]
        obs = cv2.resize(obs, (self._width, self._height), interpolation=cv2.INTER_AREA)
        obs = obs[:, :, np.newaxis]

        self.history.append(obs)
        if len(self.history) > self.nstacks:
            self.history = self.history[1:]

    def _get_stacked_obs(self):
        stack = np.concatenate(self.history, axis=2).astype(np.float32) / 255.0
        return stack

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._push_raw_obs(obs)
        info["raw_obs"] = obs
        return self._get_stacked_obs(), reward, done, info

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.nstacks):
            self._push_raw_obs(obs)
        return self._get_stacked_obs()


class DiscretizeActionWrapper(gym.Wrapper):

    def __init__(self, env, bins=10):
        """
        Convert continious action space into discrete.
        """
        super().__init__(env)
        self.env = env

        assert isinstance(env.action_space, gym.spaces.Box)
        assert len(env.action_space.shape) == 1

        dims = env.action_space[0]

        self.action_map = []

        spans = [np.linspace(env.action_space.low[d], env.action_space.high[d], bins) for d in range(dims)]

        self.action_map = list(itertools.product(*spans))

        self.action_space = gym.spaces.Discrete(len(self.action_map))

    def step(self, action):
        return self.env.step(self.action_map[action])


"""
------------------------------------------------------------------------------------------------------------------------
    Utilities
------------------------------------------------------------------------------------------------------------------------
"""


def make_environment(env_name):
    """ Construct environment of given name, including any required """
    env = gym.make(env_name)
    if "Pong" in env_name:
        env = AtariWrapper(env)
        env = NormalizeRewardWrapper(env)
    elif "CartPole" in env_name:
        env = NormalizeObservationWrapper(env)
    if isinstance(env.action_space, gym.spaces.Box):
        env = DiscretizeActionWrapper(env)
    return env


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
    p /= p.sum()  # probs are sometimes off by a little due to precision error
    return np.random.choice(range(len(p)), p=p)


def write_cv2_video(filename, frames):
    height, width, channels = frames[0].shape

    # print("writing video "+filename+" "+str(frames[0].shape)+" "+str(len(frames)))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height), isColor=True)

    for frame in frames:
        out.write(frame)

    out.release()


class MLPModel(nn.Module):
    """ A very simple Multi Layer Perceptron """

    def __init__(self, input_dims, actions):
        super(MLPModel, self).__init__()
        self.actions = actions
        self.d = prod(input_dims)
        self.fc1 = nn.Linear(self.d, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_policy = nn.Linear(64, actions)
        self.fc_value = nn.Linear(64, 1)
        self.to(DEVICE)

    def forward(self, x):
        x = torch.from_numpy(x).float().to(DEVICE)
        x = x.reshape(-1, self.d)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

    def policy(self, x):
        x = self.forward(x)
        x = self.fc_policy(x)
        x = F.softmax(x, dim=1)
        return x

    def value(self, x):
        x = self.forward(x)
        x = self.fc_value(x)
        return x


class CNNModel(nn.Module):
    """ Nature paper inspired CNN """

    def __init__(self, input_dims, actions):
        super(CNNModel, self).__init__()
        self.actions = actions
        h, w, c = input_dims
        self.conv1 = nn.Conv2d(c, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        d = prod((32, 9, 9))
        self.fc = nn.Linear(d, 256)
        self.fc_policy = nn.Linear(256, actions)
        self.fc_value = nn.Linear(256, 1)
        self.to(DEVICE)

    def forward(self, x):
        # need NCHW, but input is HWC
        if len(x.shape) == 3:
            # make a batch of 1 for a single example.
            x = x[np.newaxis, :, :, :]
        x = np.swapaxes(x, 1, 3)
        x = torch.from_numpy(x).float().to(DEVICE)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc(x.view(-1)))
        return x

    def policy(self, x):
        x = self.forward(x)
        x = self.fc_policy(x)
        x = F.softmax(x, dim=1)
        return x

    def value(self, x):
        x = self.forward(x)
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
        y = (1 - alpha) * x + (alpha) * y
        results.append(y)
    return results


def export_video(filename, frames, scale=4):
    if len(frames) == 0:
        return

    height, width, channels = frames[0].shape

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

    write_cv2_video(filename, processed_frames)


def safe_mean(X):
    return np.mean(X) if len(X) > 0 else 0

def update_params(optimizer, epochs, n_batches, agent_history, agent_advantage, epsilon, gamma, model, VF_coef,
                  ent_bonus, training_log, score_history, len_history):

    mini_batch_size = len(agent_advantage) // n_batches

    for i in range(epochs):

        ordering = list(range(len(agent_history)))
        np.random.shuffle(ordering)
        agent_history = np.array([agent_history[i] for i in ordering])
        agent_advantage = np.array([agent_advantage[i] for i in ordering])

        agent_advantage = (agent_advantage - agent_advantage.mean()) / (agent_advantage.std() + 1e-8)

        reward_sum = 0

        total_loss = 0
        total_loss_clip = 0
        total_loss_value = 0
        total_loss_entropy = 0
        total_reward_sum = 0

        for j in range(n_batches):

            optimizer.zero_grad()
            # compute losses
            loss_clip = 0.0
            loss_value = 0.0
            loss_entropy = 0.0

            batch_start = j * mini_batch_size
            batch_end = (j + 1) * mini_batch_size

            batch_sample = range(batch_start, batch_end)

            for sample in batch_sample:
                prev_observation, action, observation, reward, value_estimate, probs, isterminal = agent_history[
                    sample]
                advantage = agent_advantage[sample]

                forward = model.policy(prev_observation)[0]

                ratio = (forward[action] / probs[action])
                loss_clip += torch.min(ratio * advantage, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage)

                # td lambda for this would be better...
                target = reward if isterminal else reward + gamma * model.value(observation)[0].detach()
                pred_value = model.value(prev_observation)[0]
                td = target - pred_value

                loss_value -= VF_coef * (td * td)
                loss_entropy += ent_bonus * entropy(forward)
                reward_sum += reward

            loss = -(loss_clip + loss_value + loss_entropy) / mini_batch_size  # gradient ascent.

            loss.backward()
            optimizer.step()

            total_loss_clip += loss_clip
            total_loss_value += loss_value
            total_loss_entropy += loss_entropy
            total_loss += loss
            total_reward_sum += reward_sum

        history_string = "{}".format(
            [round(float(x), 2) for x in score_history[-5:]]
        )

        training_log.append(
            (float(total_loss),
             float(total_loss_clip),
             float(total_loss_value),
             float(total_loss_entropy),
             safe_mean(score_history[-100:]),
             safe_mean(len_history[-100:]),
             history_string)
        )


def update_params_batch(optimizer, epochs, n_batches, agent_history, agent_advantage, epsilon, gamma, model, VF_coef,
                  ent_bonus, training_log, score_history, len_history):

    mini_batch_size = len(agent_advantage) // n_batches

    # put data in tensors
    prev_observations = np.asarray(x[0] for x in agent_history)
    actions = np.asarray(x[1] for x in agent_history)
    next_observations = np.asarray(x[2] for x in agent_history)
    rewards = np.asarray(x[3] for x in agent_history)
    value_estimates = np.asarray(x[4] for x in agent_history)
    policy_probs = np.asarray(x[5] for x in agent_history)
    terminals = np.asarray(x[6] for x in agent_history)
    advantages = np.asarray(agent_advantage)

    for i in range(epochs):

        ordering = list(range(len(agent_history)))
        np.random.shuffle(ordering)

        for x in [prev_observations, actions, next_observations, rewards, value_estimates, policy_probs, terminals, advantages]:
            x = x[ordering]

        agent_history = np.array([agent_history[i] for i in ordering])
        agent_advantage = np.array([agent_advantage[i] for i in ordering])

        agent_advantage = (agent_advantage - agent_advantage.mean()) / (agent_advantage.std() + 1e-8)


        reward_sum = 0

        total_loss = 0
        total_loss_clip = 0
        total_loss_value = 0
        total_loss_entropy = 0
        total_reward_sum = 0

        for j in range(n_batches):

            optimizer.zero_grad()

            batch_start = j * mini_batch_size
            batch_end = (j + 1) * mini_batch_size

            # collect sample
            prev_observation = agent_history[batch_start:batch_end]

            for sample in batch_sample:
                prev_observation, action, observation, reward, value_estimate, probs, isterminal = agent_history[
                    sample]
                advantage = agent_advantage[sample]

                forward = model.policy(prev_observation)[0]

                ratio = (forward[action] / probs[action])
                loss_clip += torch.min(ratio * advantage, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage)

                # td lambda for this would be better...
                target = reward if isterminal else reward + gamma * model.value(observation)[0].detach()
                pred_value = model.value(prev_observation)[0]
                td = target - pred_value

                loss_value -= VF_coef * (td * td)
                loss_entropy += ent_bonus * entropy(forward)
                reward_sum += reward

            loss = -(loss_clip + loss_value + loss_entropy) / mini_batch_size  # gradient ascent.

            loss.backward()
            optimizer.step()

            total_loss_clip += loss_clip
            total_loss_value += loss_value
            total_loss_entropy += loss_entropy
            total_loss += loss
            total_reward_sum += reward_sum

        history_string = "{}".format(
            [round(float(x), 2) for x in score_history[-5:]]
        )

        training_log.append(
            (float(total_loss),
             float(total_loss_clip),
             float(total_loss_value),
             float(total_loss_entropy),
             safe_mean(score_history[-100:]),
             safe_mean(len_history[-100:]),
             history_string)
        )

def train_minibatch(model, optimizer, gamma, epsilon, VF_coef, ent_bonus, prev_states, actions, next_states, rewards, value_estimates, policy_probs, terminals, advantages):

    loss_clip = 0
    loss_value = 0
    loss_entropy = 0
    reward_sum = 0

    mini_batch_size = len(prev_states)

    for prev_state, action, state, reward, value_estimate, probs, is_terminal, advantage in zip(
            prev_states, actions, next_states, rewards, value_estimates, policy_probs, terminals, advantages):

        forward = model.policy(prev_state)[0]

        ratio = (forward[action] / probs[action])
        loss_clip += torch.min(ratio * advantage, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage)

        # also precompute targets...
        # td lambda for this would be better...
        target = reward if is_terminal else reward + gamma * model.value(state)[0].detach()
        pred_value = model.value(prev_state)[0]
        td = target - pred_value

        loss_value -= VF_coef * (td * td)
        loss_entropy += ent_bonus * entropy(forward)
        reward_sum += reward

    loss = -(loss_clip + loss_value + loss_entropy) / mini_batch_size  # gradient ascent.

    loss.backward()
    optimizer.step()

    return (float(x) for x in [loss, loss_clip, loss_value, loss_entropy, reward_sum])


def train(env_name, model: nn.Module):
    trace("Training started.")

    n_steps = 128  # steps per update
    gamma = 0.99  # discount
    lamb = 0.95  # GAE parameter
    n_batches = 4
    epsilon = 0.1
    VF_coef = 1.0  # how much loss to take from value function
    agents = 8
    epochs = 3
    ent_bonus = 0.01
    alpha = 2.5e-4

    mini_batch_size = (n_steps * agents) // n_batches

    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

    envs = [make_environment(env_name) for _ in range(agents)]

    # initialize agent
    states = [env.reset() for env in envs]

    episode_score = [0 for _ in envs]
    episode_len = [0 for _ in envs]

    training_log = []

    score_history = []
    len_history = []

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

        start_time = time.time()

        agent_history = []
        agent_advantage = []

        start_time = time.time()

        for i, env in enumerate(envs):

            state = states[i]
            history = []

            for t in range(n_steps):

                probs = model.policy(state)[0].detach().cpu()
                action = sample_action(probs)
                value_estimate = model.value(state)[0].detach().cpu()

                prev_state = state.copy()

                state, reward, done, info = env.step(action)

                raw_reward = info.get("raw_reward", reward)

                history.append((prev_state, action, state, reward, value_estimate, probs, done))
                episode_score[i] += raw_reward
                episode_len[i] += 1

                if done:
                    _ = env.reset()
                    score_history.append(episode_score[i])
                    len_history.append(episode_len[i])
                    episode_score[i] = 0
                    episode_len[i] = 0
                    if i == 0:
                        export_video("video {}.mp4".format(step), video_frames)
                        video_frames = []

            states[i] = state

            # generate advantage estimates
            # note, this can be done much faster, but this will do for the moment.

            advantages = []

            """
            for t in range(nsteps):
                advantage_t = history[t][3] - history[t][4] 

                factor = 1

                for i in range(t+1, nsteps-1):
                    _, _, _, reward_i, value_i, _, terminal_i = history[i]
                    _, _, _, _, value_next_i, _, terminal_next_i = history[i+1]

                    factor *= (gamma)
                    advantage_t += factor * reward_i

                    if terminal_i:
                        break
                else:
                    advantage_t += factor * value_i

                advantages.append(advantage_t)
            """

            for t in range(n_steps):
                advantage_t = 0
                for i in range(t, n_steps - 1):
                    _, _, _, reward_i, value_i, _, terminal_i = history[i]
                    _, _, _, _, value_next_i, _, terminal_next_i = history[i + 1]

                    if terminal_next_i:
                        value_next_i = 0

                    td_i = reward_i + (gamma * value_next_i) - value_i
                    advantage_t += ((lamb * gamma) ** (i - t)) * td_i

                    if terminal_next_i:
                        break
                advantages.append(advantage_t)

            agent_history.extend(history)
            agent_advantage.extend(advantages)

        step_time = time.time() - start_time

        agent_history = np.array(agent_history)
        agent_advantage = np.array(agent_advantage)

        # normalize advantages
        agent_advantage = (agent_advantage - agent_advantage.mean()) / (agent_advantage.std() + 1e-8)

        # export video preview of agent.
        # export_video("preview.mp4", video_frames)

        total_loss_clip = 0
        total_loss_value = 0
        total_loss_entropy = 0
        total_loss = 0
        total_reward_sum = 0

        for i in range(epochs):

            ordering = list(range(len(agent_history)))
            np.random.shuffle(ordering)
            agent_history = np.array([agent_history[i] for i in ordering])
            agent_advantage = np.array([agent_advantage[i] for i in ordering])

            for j in range(n_batches):

                # put together a minibatch.
                batch_start = j * mini_batch_size
                batch_end = (j + 1) * mini_batch_size
                slices = (x[batch_start:batch_end] for x in [*zip(*agent_history), agent_advantage])

                loss, loss_clip, loss_value, loss_entropy, reward_sum = train_minibatch(model, optimizer, gamma, epsilon, VF_coef, ent_bonus, *slices)

                total_loss_clip += loss_clip
                total_loss_value += loss_value
                total_loss_entropy += loss_entropy
                total_loss += loss
                total_reward_sum += reward_sum

            history_string = "{}".format(
                [round(float(x), 2) for x in score_history[-5:]]
            )

            training_log.append(
                (float(total_loss),
                 float(total_loss_clip),
                 float(total_loss_value),
                 float(total_loss_entropy),
                 safe_mean(score_history[-100:]),
                 safe_mean(len_history[-100:]),
                 history_string)
            )

        step_time = (time.time() - start_time) / (n_steps * agents)

        if step == 0:
            print("Training at {:.1f}fps".format(1.0 / step_time))

        if step % 100 == 0:
            print("{:<11}{:<11}{:<11}{:<11}{:<11}{:<11}{:<11}".format("step", "loss", "loss_clip", "loss_value",
                                                                      "loss_ent", "ep_score", "ep_len"))
        if step % 10 == 0:
            print("{:<11}{:<11.3f}{:<11.3f}{:<11.3f}{:<11.3f}{:<11.3f}{:<11.3f}{:<11}".format(
                str(step) + " " + str(step * n_steps * agents // 1000) + "K",
                training_log[-1][0],
                training_log[-1][1],
                training_log[-1][2],
                training_log[-1][3],
                training_log[-1][4],
                training_log[-1][5],
                training_log[-1][6]
            ))
            # print([float(x) for x in [target, pred_value]])

        """
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
        """

    return training_log


def run_experiment(env_name, Model):

    env = make_environment(env_name)
    n_actions = env.action_space.n
    obs_space = env.observation_space.shape

    print("Playing {} with {} obs space and {} actions.".format(env_name, obs_space, n_actions))

    model = Model(obs_space, n_actions)
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
    show_cuda_info()
    #run_experiment("Pong-v4", CNNModel)
    run_experiment("CartPole-v0", MLPModel)

