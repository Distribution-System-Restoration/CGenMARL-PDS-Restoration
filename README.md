# Soft Actor-Critic Multi-Agent Reinforcement Learning for Load Restoration in Smart Grid Systems

This project focuses on optimizing load restoration strategies using the Soft Actor-Critic (SAC) algorithm within a decentralized actor-centralized critic (DACA) framework.

# Project Overview
In this project, we address the problem of load restoration in power distribution systems after a fault occurs. We divide the distribution system into different microgrids, each managed by an independent agent. These agents operate without communication and observe their own circuit breakers. The overall goal is to maximize the restored load under defined constraints using limited solar power distributed throughout the system.

# Methodology
We employ the Soft Actor-Critic (SAC) algorithm to manage the load restoration process. Each agent has its own actor network, while a common critic network is shared among all agents. This DACA approach allows for efficient learning and decision-making in a multi-agent environment. The training process leverages Temporal Difference (TD) error to update the critic network and guide the actors' policies.

# Key Features
* Decentralized Actor-Centralized Critic (DACA) Framework: Individual actors for each agent with a shared critic for effective coordination.
* Soft Actor-Critic (SAC) Algorithm: Enhances the robustness and performance of the load restoration strategy.
* Simulation Environment: Implemented using Python and the opendssdirect library to connect with OpenDSS for power flow simulations.
* Experimental Setup: Tested on the IEEE 123-bus system with a full load of 3025 kW and 2600 kW of available solar power during faults.
* 
# Getting Started
To get started with this project, follow the steps below:
* Prerequisites
  ** Python 3.7 or higher
  ** PyTorch
  
