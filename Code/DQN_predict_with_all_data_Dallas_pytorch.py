import argparse
import os
import pandas as pd
import regex as re
import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.python.client import device_lib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('--output_file_name', default="rewards", type=str, help='Output Filename to store the rewards.')

args = parser.parse_args()
print('Argument parser inputs', args)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.config.list_physical_devices('GPU')
# print(device_lib.list_local_devices())

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

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

# The following 3 lines were added for the Dallas dataset
data['Source'] = data['Source_X'].astype(str) + ', ' + data['Source_Y'].astype(str)
data['Target'] = data['Target_X'].astype(str) + ', ' + data['Target_Y'].astype(str)
data.drop_duplicates('Target', inplace=True)

data1 = data.copy()
data1.drop(['Path Type', 'Obstacle', 'Path ID', 'Channel Coefficient',
            'Source_X', 'Source_Y', 'Source_Z', 'Target_Z', 'Source', 'Target'], axis = 1, inplace = True)
data1 = data1.astype(float)
# Display the first 10 rows
print(data1.head(10))

min_x = min(data['Target_X'])
min_y = min(data['Target_Y'])
max_x = max(data['Target_X'])
max_y = max(data['Target_Y'])
start_x, start_y = min_x, min_y
end_x, end_y = max_x, max_y

print("Starting position = ", start_x, start_y)
print("Ending position = ", end_x, end_y)

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
    if next_x == max_x and next_y == max_y:
        return 5000, next_x, next_y

    # Position inside the grid other inside the boundaries
    if next_x >= min_x and next_x <= max_x and next_y >= min_y and next_y <= max_y:
        return r, next_x, next_y

    # If the agent hits the wall either the x cooridinate will increase or decrease or the y coordinate will increase or decrease
    if next_x < min_x or next_x > max_x or next_y < min_y or next_y > max_y:
        return -5000, cur_x, cur_y

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
        self.behaviour_model = DQN(self.state_size, self.action_size).to(device)
        print("Summary of behaviour model = ", summary(self.behaviour_model, (9,)))


    def load_model(self):
        self.behaviour_model.load_state_dict(torch.load(args.output_file_name, weights_only=True))

    def predict(self, cur_x, cur_y):

        rewards = []
        path = [[cur_x, cur_y]]
        print("Starting State = ", cur_x, cur_y)

        while ((cur_x, cur_y) != (end_x, end_y)):

            current_input_to_model = torch.tensor(data1[data1['Target_X'] == cur_x][data1['Target_Y'] == cur_y].values.astype('float32'), device = device)

            # Predicting the action using the behaviour model.
            with torch.no_grad():
                action = agent.behaviour_model(current_input_to_model).max(1).indices.item()

            print("Next Action = ", action)
            reward, next_x, next_y = next_state(cur_x, cur_y, actions[action], end_x, end_y)
            print("Next State = ", next_x, next_y)
            cur_x, cur_y = next_x, next_y
            path.append([cur_x, cur_y])
            rewards.append(reward)

        return (rewards, path)



if __name__ == "__main__":

    state_size = 9
    action_size = 4
    agent = Dallas(state_size, action_size)

    batch_size = 16
    all_rewards = []
    rewards = []
    average_rewards = []
    stepcounts = []

    cur_x, cur_y = start_x, start_y

    agent.load_model()
    rewards, path = agent.predict(cur_x, cur_y)

    # with open(args.output_file_name+'rewards_predicted', 'w') as f:
    #
    #     # using csv.writer method from CSV package
    #     write = csv.writer(f)
    #
    #     write.writerows(rewards)

    plt.plot(rewards, label = 'rewards')
    plt.xlabel('Steps', fontsize=20)
    plt.ylabel('Rewards', fontsize=20)
    plt.title('Rewards vs Steps', fontsize=20)
    plt.legend(fontsize=15)

    plt.savefig(args.output_file_name+'predicted_rewards')
    plt.pause(1)
    plt.close()

    x, y = zip(*path)
    plt.scatter(x,y, label='path')
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.xlabel('X Coordinate', fontsize=20)
    plt.ylabel('Y Coordinate', fontsize=20)
    plt.title('Path taken by the agent: ', fontsize=20)
    plt.legend(fontsize=15)

    plt.savefig(args.output_file_name + 'path_taken')
    plt.pause(1)
    plt.close()
