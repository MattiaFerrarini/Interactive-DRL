from collections import deque, namedtuple
import time
import environment
import qNetwork
import numpy as np
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt

# training variables
MEMORY_SIZE = 100_000       # size of memory buffer
STEPS_FOR_UPDATE = 4        # perform a learning update every C time steps
MINIBATCH_SIZE = 64         # minimum number of examples for q_network update
MIN_EPSILON = 0.01          # smalles allowed epsilon
EPSILON_DECAY = 0.995       # epsilon decay after every episode
STATE_SIZE = 6              # size of np array representing the state
NUM_ACTIONS = 4             # number of possible acctions (up, down, left, right)
MAX_EPISODES = 5000         # maximum number of training episodes
MAX_TIMESTEPS = 2000        # maximum number of steps per episode
EPISODES_TO_AVG = 100       # episodes to average in order to determine the agent's level
SCORE_FOR_SUCCESS = 200     # averaging this score over the last EPISODES_TO_AVG will stop training

# memory buffer
memory_buffer = deque(maxlen=MEMORY_SIZE)
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "game_over"])

# epsilon for epsilon-greedy policy
epsilon = 1.0

# variables to keep track of agent's improvement
total_reward_history = []


'''
Update epsilon based on EPSILON_DECAY and MIN_EPSILON.
'''
def update_epsilon():
    global epsilon
    epsilon = max(MIN_EPSILON, EPSILON_DECAY*epsilon)


'''
Returns an action according to an epsilon-greedy policy.
'''
def get_action(q_values, epsilon=0.0):
    if random.random() > epsilon:
        return np.argmax(q_values.detach().numpy()[0])
    else:
        return random.choice(np.arange(NUM_ACTIONS))


'''
Returns a subset of experiences that can be used for model updates.
'''
def select_experiences(memory_buffer):
    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    separated = []
    for i in range(5):
        separated.append(torch.tensor(np.array([e[i] for e in experiences if e is not None]), dtype=torch.float32))
    return separated


'''
Plots the per-episode and mean rewards over the training process.
'''
def plot_history(history):
    window_size = (len(history) * 10) // 100
    # Use Pandas to calculate the rolling mean (moving average).
    rolling_mean = pd.DataFrame(history).rolling(window_size).mean()

    # plot per-episode rewards and rolling mean
    plt.plot(history, label='Reward per episode')
    plt.plot(rolling_mean, label=f'Rolling Mean ({window_size} episodes)')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


'''
Trains the agent.
'''
def train(rows, cols):
    # initialize the environment
    env = environment.Env(rows, cols)

    # initialize the models
    q_net = qNetwork.QNetwork(STATE_SIZE, NUM_ACTIONS)
    t_net = qNetwork.QNetwork(STATE_SIZE, NUM_ACTIONS)
    t_net.load_state_dict(q_net.state_dict())
    q_net.init_optimizer()

    for i in range(MAX_EPISODES):
    
        # ignore events to prevent crash
        env.ignore_events()

        # reset the environment to the initial state and get the initial state
        state = env.reset()
        total_reward = 0
    
        for t in range(MAX_TIMESTEPS):
            # render environment
            env.render()

            # choose action
            q_state = np.expand_dims(state, axis=0)
            q_values = q_net(torch.tensor(q_state, dtype=torch.float32))
            action = get_action(q_values, epsilon)

            # take the computed action
            next_state, reward, game_over = env.step(action)

            # remember the current experience by storing it into the memory buffer
            memory_buffer.append(experience(state, action, reward, next_state, game_over))

            # update the models every STEPS_FOR_UPDATE steps
            if (t+1) % STEPS_FOR_UPDATE == 0 and len(memory_buffer) >= MINIBATCH_SIZE:
                qNetwork.updateModels(q_net, t_net, select_experiences(memory_buffer))

            # update variables
            state = next_state.copy()
            total_reward += reward
        
            # if the episode ended, there are no more steps to take
            if game_over:
                break

        # update epsilon
        update_epsilon()

        # save progress and print it
        total_reward_history.append(total_reward)
        last_episodes_mean = np.mean(total_reward_history[-EPISODES_TO_AVG:])

        print(f"\rEpisode {i+1} | Total point average of the last {EPISODES_TO_AVG} episodes: {last_episodes_mean:.2f}", end="")
        if (i+1) % EPISODES_TO_AVG == 0:
            print(f"\rEpisode {i+1} | Total point average of the last {EPISODES_TO_AVG} episodes: {last_episodes_mean:.2f}")

        # if the desired mean score has been reached, training can be stopped
        if last_episodes_mean >= SCORE_FOR_SUCCESS:
            print(f"\n\nEnvironment solved in {i+1} episodes!")
            break
        
    # save the trained model parameters
    q_net.save('trained_agent.pt')


'''
Trains the model and displays the training history
'''
if __name__ == "__main__":

    rows, cols = input("Insert the number of rows and columns separated by a space: ").split()
    
    # train the agent. Keep track of the training time
    start_time = time.time()
    train(int(rows), int(cols))
    tot_time = time.time() - start_time

    # print training time
    print(f"Total training time: {tot_time:.2f}s ({(tot_time/60):.2f} min)")

    # plot total reward history
    plot_history(total_reward_history)