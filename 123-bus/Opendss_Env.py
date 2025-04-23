import numpy as np
import torch
import opendssdirect as dss
from collections import deque
import copy

class World:
    def __init__(self, args, agent_n, root_path):
        
        self.root_path = root_path
        dss.Command('clearall')
        dss.Text.Command(f'Redirect "{self.root_path}/CGAN-SAC/123Bus/123Bus/IEEE123Master.dss"') 
        self.agent_n = agent_n
        seed = 128
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.weights = self.random_weight_generator(91)
        self.num_areas = args.N
        self.P_gen = args.P_gen
        self.M = args.M
        
        self.regions_switches = [
            ['L1', 'L2', 'L8', 'L9', 'L12', 'L18', 'L21', 'L23', 'L25', 'L26'],
            ['L35', 'L40', 'L42', 'L44', 'L45'],
            ['L57', 'L60', 'L61'],
            ['L66', 'L72', 'L76', 'L77', 'L96'],
            ['L100', 'L104', 'L107']
        ]
        
        self.states = [[0 for _ in range(len(sublist))] for sublist in self.regions_switches]
        self.base_reward = self.get_load_data()
        print("base reward is: ", self.base_reward)
        
    def get_base_power(self):
        self.reset_world()
        return self.get_restored_power()

    def reset_network(self):
        self.previous_reward = self.base_reward
        dss.Command('clearall')
        dss.Text.Command(f'Redirect "{self.root_path}/CGAN-SAC/123Bus/123Bus/IEEE123Master.dss"') 
        for region in self.regions_switches:
            for switch_name in region:
                self.open_switch(switch_name)      
        dss.Solution.Solve()


    def reset_world(self):
        self.done = 0
        self.states = [[0 for _ in range(len(sublist))] for sublist in self.regions_switches]
        self.reset_network()
        return self.states
    
    def get_restored_power(self):
        dss.Solution.Solve()
        powers = []
        for load in dss.Loads:
            powers.append(dss.CktElement.TotalPowers()[0])
        return sum(powers)
        
    def get_status(self, area_idx):
        if area_idx >= self.num_areas:
            raise ValueError("Invalid area index")
        return self.states[area_idx]
    
    def get_done(self):
        return self.done
    
    def get_reward(self):
        current_rew = self.get_load_data()
        restored = self.get_restored_power()
        delta_rew = current_rew - self.previous_reward
        if restored > self.P_gen:
            rew = -self.M*(restored - self.P_gen)**2
            self.done=1
        else:
            rew = delta_rew 
        self.previous_reward = current_rew
        return rew, current_rew
    

    def step(self, action_n, step):
        obs_next_n = []
        reward_n = []
        done_n = []
        restored_n = []
        for area_idx, action in enumerate(action_n):
            self.switching(area_idx, action) 
        reward, restored = self.get_reward()
        for agent_idx, agent in enumerate(self.agent_n):
            reward_n.append(reward)
            obs_next_n.append(self.get_status(agent_idx))
            done_n.append(self.get_done())
            restored_n.append(restored)
        return obs_next_n, reward_n, done_n, restored_n

    
    def open_switch(self, switch_name):
        dss.run_command(f"open Line.{switch_name} term=2")
        dss.Solution.Solve()

    def close_switch(self, switch_name):
        dss.run_command(f"close Line.{switch_name} term=2")
        dss.Solution.Solve()
                
        
    def switching(self, area_idx, action):
        num_switches = len(self.states[area_idx])
        if len(action) != num_switches*3:
            raise ValueError(f"Mismatch in number of switches for area {area_idx}: expected {num_switches*3}, got {len(action)}")
        switch_on_actions = np.array(action[:num_switches], dtype=int)
        switch_off_actions = np.array(action[num_switches:num_switches*2], dtype=int)
        no_action = np.array(action[num_switches*2:], dtype=int)
        if not any(no_action):
            for switch_idx, (on_action,off_action) in enumerate(zip(switch_on_actions,switch_off_actions)):
                    switch_name = self.regions_switches[area_idx][switch_idx]
                    if on_action: 
                        self.close_switch(switch_name)
                        self.states[area_idx][switch_idx] |= 1
                    if off_action:  
                        self.open_switch(switch_name)
                        self.states[area_idx][switch_idx] &= 0
 

    def get_load_data(self):
        dss.Solution.Solve()
        powers = []
        for load in dss.Loads:
            powers.append(dss.CktElement.TotalPowers()[0])
        if self.weights is None:
            self.weights = [1] * len(powers)
        if len(self.weights) != len(powers):
            raise ValueError("Length of weights must match the number of power values")
        weighted_powers = [p * w for p, w in zip(powers, self.weights)]
        return sum(weighted_powers)

   
    def get_per_unit_v(self):
        dss.Solution.Solve()
        return dss.Circuit.AllBusMagPu()
    
    
    def random_weight_generator(self, num_switches):
        numbers_0_to_10 = torch.arange(0, 11)
        remaining_numbers = torch.randint(0, 11, (num_switches - len(numbers_0_to_10),))
        combined_numbers = torch.cat((numbers_0_to_10, remaining_numbers))
        shuffled_numbers = combined_numbers[torch.randperm(combined_numbers.size(0))]
        return shuffled_numbers