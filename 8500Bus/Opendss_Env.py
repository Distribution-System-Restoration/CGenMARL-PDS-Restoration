import numpy as np
import torch
import opendssdirect as dss
from collections import deque
import copy

class World:
    def __init__(self, args, agent_n, root_path):
        
        self.root_path = root_path
        dss.Command('clearall')
        dss.Text.Command(f'Redirect "{self.root_path}/CGAN-SAC/8500Bus/8500-Node/Master.dss"') 
        self.agent_n = agent_n
        seed = 128
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.weights = self.random_weight_generator(1177)
        self.num_areas = args.N
        self.P_gen = args.P_gen
        self.M = args.M
        selected_switches = self.randomly_select_switches(num_switches=sum(args.obs_dim_n))
        regions_switches_n = self.group_switches_into_regions(selected_switches, num_regions=self.num_areas)
        self.regions_switches = self.assign_regions_to_agents(regions_switches_n, num_agents=self.num_areas)
        self.states = [[0 for _ in range(len(sublist))] for sublist in self.regions_switches]
        self.base_reward = self.get_load_data()
        print("base reward is: ", self.base_reward)
        
    def get_base_power(self):
        self.reset_world()
        return self.get_restored_power()

    def reset_network(self):
        self.previous_reward = self.base_reward
        dss.Command('clearall')
        dss.Text.Command(f'Redirect "{self.root_path}/CGAN-SAC/8500Bus/8500-Node/Master.dss"') 
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

    
    def get_all_lines(self):
        """Retrieve all lines in the system."""
        dss.Lines.First()
        all_lines = []
        while True:
            line_name = dss.CktElement.Name()
            all_lines.append(line_name.split('.')[-1])  # Get the name of the line
            if not dss.Lines.Next():
                break
        return all_lines

    def randomly_select_switches(self, num_switches=100):
        """Randomly select lines to act as switches."""
        all_lines = self.get_all_lines()
        if len(all_lines) < num_switches:
            raise ValueError("Not enough lines in the system to select as switches.")
        selected_switches = np.random.choice(all_lines, num_switches, replace=False).tolist()
        return selected_switches

    def group_switches_into_regions(self, selected_switches, num_regions=10):
        """Group selected switches into regions."""
        region_size = len(selected_switches) // num_regions
        regions = [
            selected_switches[i * region_size:(i + 1) * region_size]
            for i in range(num_regions)
        ]
        return regions

    def assign_regions_to_agents(self, regions, num_agents=10):
        """Assign regions to agents."""
        if len(regions) < num_agents:
            raise ValueError("Not enough regions for the number of agents.")
        agent_regions = [regions[i] for i in range(num_agents)]
        return agent_regions

