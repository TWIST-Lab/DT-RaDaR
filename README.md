# DT-RaDaR

This codebase facilitates the simulation of the experiments done for the DT-RaDaR framework.

## Introduction

This experiment introduces the novel framework for robot navigation developed using the Deep Q-Learning algorithm on the ray tracing data collected using the SIONNA RT and Digital Twin. The Digital Twins were created for the Dallas and the Houston cities. The RT_Data folder contains the ray tracing data collected for the Dallas and the Houston cities using the SIONNA RT inside the Digital Twin. The Code data contains the python files for performing the DQN training for robot navigation and also the for performing the inference on the ray tracing data for the robot navigation on the digital twins of the Dallas and the Houston cities. 

## Files

The DQN algorithm was created using the pytorch library.
The DQN_with_all_data_obstacle_Dallas_pytorch.py and DQN_with_all_data_obstacle_Houston_pytorch.py files contains the DQN code used for training on the Dallas and Houston cities' digital twins respectively. 
The DQN_predict_with_all_data_Dallas_pytorch.py and DQN_predict_with_all_data_obstacle_Houston_pytorch.py files contains the code used for performing the inference on the Dallas and Houston cities' digital twins respectively.

## Setup and Prerequisites

### Prerequisites

Before running the MARL.py file make sure you have the following installed.

- Ubuntu OS or Windows
- Python3 - [Download](https://www.python.org/downloads/)
- Pytorch - To install the pytorch after installing the python3, run the following command --> pip3 install torch torchvision torchaudio

### Getting Started

Once you have successfully installed all the necessary dependencies, proceed to run the python file. For runnuing the python file see the instructions below.

### Instructions

1. Put the corresponding python file and the data file in the same folder of the respective digital twin city in the same folder.

2. Then run the python file using the following command --> python3 (filename).py

3. Every python file takes the following commandline arguments:

   a. --output_file_name, default="rewards"
   
   b. --episodes, default=10

   c. --id_gpu, default=0

   d. --epochs, default=5

   e. --steps, default=50 

## Cite

If you find this simulation framework useful in your research, please cite our work:

```Sunday Amatare, Gaurav Singh, Raul Shakya, Aavash Kharel, Ahmed Alkhateeb and Debashri Roy, "DT-RaDaR: Digital Twin Assisted Robot Navigation using Differential Ray-Tracing", Available at SSRN: https://arxiv.org/html/2411.12284v1```

## Acknowledgments

Thank you for choosing the DT-RaDaR framework. For any questions, feedback, or collaboration opportunities, please feel free to reach out.
