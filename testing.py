import time
from environment import Env
import numpy as np
from qNetwork import QNetwork
import torch

# the amount of seconds between agent's moves
TIME_FOR_UPDATE = 0.5


if __name__ == '__main__':

    rows, cols = input("Insert the number of rows and columns separated by a space: ").split()

    # init the environmet
    env = Env(int(rows), int(cols))

    # load the agent's model
    q_net = QNetwork(state_size=6, num_actions=4)
    q_net.load_state_dict(torch.load('trained_agent.pt'))
    q_net.eval()

    # this allows the user to customize the grid by choosing the obstacles' positions
    env.create_custom_env()
    state = env.reset(custom_env=True)

    # variables for session status and stats
    game_over = False
    total_reward = 0
    move_count = 0

    while not game_over:

        # ignore events to prevent crash
        env.ignore_events()

        # render the environmet
        env.render()

        # choose an action
        q_state = np.expand_dims(state, axis=0)
        q_values = q_net(torch.tensor(q_state, dtype=torch.float32))
        action = np.argmax(q_values.detach().numpy()[0])

        # take the action
        state, reward, game_over = env.step(action)
        total_reward += reward
        move_count += 1

        # wait before next move
        time.sleep(TIME_FOR_UPDATE)

    print(f"The game lasted {move_count} moves and terminated with a total reward of {total_reward}.")