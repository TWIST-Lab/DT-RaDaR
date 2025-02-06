import argparse
import os
import pandas as pd
import regex as re
import matplotlib.pyplot as plt
import time
# import tensorflow as tf
from collections import deque
import numpy as np
import random
import csv
# from tensorflow.python.client import device_lib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim


parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('--id_gpu', default=0, type=int, help='which gpu to use.')
# parser.add_argument('--data_folder', help='Location of the data directory', type=str)
parser.add_argument('--episodes', default=10, type=int, help='how many episodes to run.')
parser.add_argument('--epochs', default=5, type=int, help='how many epochs to run for.')
parser.add_argument('--steps', default=50, type=int, help='how many steps to run for.')
parser.add_argument('--output_file_name', default="rewards", type=str, help='Output Filename to store the rewards.')

args = parser.parse_args()
print('Argument parser inputs', args)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# print(tf.config.list_physical_devices('GPU'))
# print(device_lib.list_local_devices())

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print("Device = ", device)

if device == torch.device("cuda"):
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

data = pd.read_csv("dallas-data-merged.csv")
data.drop_duplicates(inplace = True)
def extract_coordinates(coord_str):
    pattern = r'-?\d+\.\d*'
    coordinates = re.findall(pattern, coord_str)
    return list(map(float, coordinates))

# Apply the function to create separate columns
data['Source_X'], data['Source_Y'], data['Source_Z'] = zip(*data['Source Coordinates'].apply(extract_coordinates))
data['Target_X'], data['Target_Y'], data['Target_Z'] = zip(*data['Target Coordinate'].apply(extract_coordinates))

# Drop the original coordinate columns
data.drop(columns=['Source Coordinates', 'Target Coordinate'], inplace=True)

real_parts = []
imaginary_parts = []

for complex_str in data['Channel Coefficient']:
    real_part = complex(complex_str).real
    imag_part = complex(complex_str).imag
    real_parts.append(real_part)
    imaginary_parts.append(imag_part)

# Add real and imaginary parts to the DataFrame
data['Real_Part(Magnitude)'] = real_parts
data['Imaginary_Part(Phase Shift)'] = imaginary_parts

data['Target'] = data['Target_X'].astype(str) + ', ' + data['Target_Y'].astype(str)
data.drop_duplicates('Target', inplace=True)

data1 = data.copy()
data1.drop(['Path Type', 'Path ID', 'Channel Coefficient', 'Target',
            'Source_X', 'Source_Y', 'Source_Z', 'Target_Z'], axis = 1, inplace = True)
data1 = data1.astype(float)
# Display the first 10 rows
print(data1.head(10))

data1 = data1.sort_values(['Target_X', 'Target_Y'], ascending = [True, True], ignore_index=True)
data2 = data1[data1['Obstacle'] == 0]

start_x = -250
start_y = -319
end_x = -50
end_y = 80

min_x = min(data1['Target_X'])
min_y = min(data1['Target_Y'])
max_x = max(data1['Target_X'])
max_y = max(data1['Target_Y'])

# Reward is just the negative of the total distance from the starting position to the target position
def cal_reward(next_x, next_y, tar_x, tar_y):
    return - ((abs(tar_x - next_x) + abs(tar_y - next_y)) )

def cur_action(action):
    cur_act = {(-1, 0): "left", (1, 0): "right", (0, 1): "down", (0, -1): "up"}
    return cur_act[action]

# Next state function to give the next state by taking action on the current state
def next_state(cur_x, cur_y, action, end_x, end_y):

    next_x = cur_x + action[0]
    next_y = cur_y + action[1]

    # Calculating the reward based on the next state
    r = cal_reward(next_x, next_y, end_x, end_y)

    # Position of the goal
    if next_x == end_x and next_y == end_y:
        return 5000, next_x, next_y

    # If the agent hits the wall either the x cooridinate will increase or decrease or the y coordinate will increase or decrease
    if next_x < min_x or next_x > max_x or next_y < min_y or next_y > max_y:
        return -5000, cur_x, cur_y

    # If next position is an obstacle known using the obstacle variable, then return negative reward and the current position
    if data1[data1['Target_X'] == next_x][data1['Target_Y'] == next_y]['Obstacle'].values[0] == 1.0:
        return -5000, cur_x, cur_y

    # Position inside the grid other inside the boundaries
    if next_x >= min_x and next_x <= max_x and next_y >= min_y and next_y <= max_y:
        return r, next_x, next_y

actions = {(0): (-1, 0), (1): (1, 0), (2): (0, -1), (3): (0, 1)}


class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_size, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class Dallas(DQN):

    def __init__(self, state_size, action_size):
        super(Dallas, self).__init__(state_size, action_size)

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.9
        self.epsilon = 0.1
        self.learning_rate = 0.001
        self.move_target_steps = 100
        self.behaviour_model = DQN(self.state_size, self.action_size).to(device)
        self.target_model = DQN(self.state_size, self.action_size).to(device)
        self.target_model.load_state_dict(self.behaviour_model.state_dict())

        self.optimizer = optim.AdamW(self.behaviour_model.parameters(), lr=self.learning_rate, amsgrad=True)

        print("Summary of behaviour model = ", summary(self.behaviour_model, (10,)))
        print("Summary of target model = ", summary(self.target_model, (10,)))

    def memorize(self, y, action, current_state):
        self.memory.append((y, action, current_state))


    def replay(self, batch_size):
        print("Training the network!......")
        start_time = time.time()
        minibatch = random.sample(self.memory, batch_size)

        batch = list(zip(*minibatch))

        batch[0] = torch.tensor(batch[0], device=device)
        batch[2] = torch.cat(batch[2])

        y_pred = self.behaviour_model(torch.tensor(batch[2]))
        y_actions = y_pred.gather(1, torch.tensor(batch[1], device=device).unsqueeze(1))

        criterion = nn.SmoothL1Loss()
        loss = criterion(y_actions, batch[0].unsqueeze(1))
        # Optimize the model
        self.optimizer.zero_grad()

        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.behaviour_model.parameters(), 100)
        self.optimizer.step()

        end_time = time.time() - start_time
        print("Time taken = ", end_time)

    def move_target(self):
        self.target_model.load_state_dict(self.behaviour_model.state_dict())




if __name__ == "__main__":

    state_size = 10
    action_size = 4
    agent = Dallas(state_size, action_size)

    batch_size = 16
    all_rewards = []
    all_average_rewards = []
    episodic_rewards = []
    episodic_average_rewards = []
    stepcounts = []

    cur_x, cur_y = start_x, start_y

    for e in range(args.episodes):

        cur_x, cur_y = start_x, start_y

        step = 0
        r_avg = 0
        rewards = []
        average_rewards = []

        # for step in range(args.steps):
        while((cur_x, cur_y) != (end_x, end_y)):

            current_input_to_model = torch.tensor(data1[data1['Target_X'] == cur_x][data1['Target_Y'] == cur_y].values.astype('float32'), device = device)

            # Taking greedy or exploratory action based on the epsilon. It uses the behaviour model.
            if np.random.uniform() > agent.epsilon:
                print("Taking greedy action..... Exploiting!!.....")

                with torch.no_grad():
                    action = agent.behaviour_model(current_input_to_model).max(1).indices.item()
                    print("Action = ", action)
            else:
                print("Taking random action..... Exploring!!.....")
                action = random.randrange(4)

            reward, next_x, next_y = next_state(cur_x, cur_y, actions[action], end_x, end_y)

            print("Absolute Reward = ", reward)
            r_avg = r_avg + (reward - r_avg)/(step + 1)
            print("Average Reward = ", r_avg)
            rewards.append(reward)
            average_rewards.append(r_avg)
            all_rewards.append(reward)
            all_average_rewards.append(r_avg)


            # It uses the target model to predict the target y for computing the error and loss
            if (next_x, next_y) != (end_x, end_y):
                print("next state = ", next_x, next_y)
                print("end state = ", end_x, end_y)

                next_input_to_model = torch.tensor(data1[data1['Target_X'] == next_x][data1['Target_Y'] == next_y].values.astype('float32'), device = device)
                with torch.no_grad():
                    y = agent.target_model(next_input_to_model).max().item()
                    print("Q next pred after predicting = ", y)

                yt = reward + agent.gamma * np.max(y)

            else:
                yt = reward

            agent.memorize(yt, action, current_input_to_model)

            cur_x, cur_y = next_x, next_y
            print("episode: {}/{}, step: {}".format(e+1, args.episodes, step+1))

            # Should the training take place at every step? Or after every particular interval of steps?
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            # Copying the behaviour model into the target model.
            if step % agent.move_target_steps == 0:
                print("Moving the target!........................ ")
                agent.move_target()

            step += 1

        stepcounts.append(step)
        episodic_rewards.append(sum(rewards) / len(rewards))
        episodic_average_rewards.append(sum(average_rewards) / len(average_rewards))


        plt.plot(rewards, label='Rewards')
        plt.xlabel('Steps', fontsize=20)
        plt.ylabel('Rewards', fontsize=20)
        plt.title('Rewards vs Steps', fontsize=20)
        plt.legend(fontsize=15)
        plt.savefig(args.output_file_name+"absolute_rewards_episode_"+str(e))

        plt.pause(1)
        plt.close()

        plt.plot(average_rewards, label='Average Rewards')
        plt.xlabel('Steps', fontsize=20)
        plt.ylabel('Rewards', fontsize=20)
        plt.title('Rewards vs Steps', fontsize=20)
        plt.legend(fontsize=15)
        plt.savefig(args.output_file_name + "average_rewards_episode_" + str(e))

        plt.pause(1)
        plt.close()

        torch.save(agent.behaviour_model.state_dict(), args.output_file_name + str(e))

    print("Maximum Reward = ", max(rewards))

    with open(args.output_file_name+'episodic_rewards', 'w') as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(episodic_rewards)

    with open(args.output_file_name+'average_episodic_rewards', 'w') as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(episodic_average_rewards)


    plt.plot(all_rewards, label = 'All Rewards')
    plt.xlabel('Steps', fontsize=20)
    plt.ylabel('Rewards', fontsize=20)
    plt.title('Rewards vs Steps', fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(args.output_file_name + "all_rewards")
    plt.pause(1)
    plt.close()


    plt.plot(all_average_rewards, label='All Average Rewards')
    plt.xlabel('Steps', fontsize=20)
    plt.ylabel('Average Rewards', fontsize=20)
    plt.title('Average Rewards vs Steps', fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(args.output_file_name + "all_average_rewards")
    plt.pause(1)
    plt.close()

    plt.plot(episodic_rewards, label='Episodic Rewards')
    plt.xlabel('Steps', fontsize=20)
    plt.ylabel('Rewards', fontsize=20)
    plt.title('Rewards vs Steps', fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(args.output_file_name + "episodic_all_rewards")
    plt.pause(1)
    plt.close()

    plt.plot(episodic_average_rewards, label='Episodic Average Rewards')
    plt.xlabel('Steps', fontsize=20)
    plt.ylabel('Average Rewards', fontsize=20)
    plt.title('Average Rewards vs Steps', fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(args.output_file_name + "episodic_average_rewards")
    plt.pause(1)
    plt.close()