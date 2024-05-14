import numpy as np

class MemoryBuffer():
    def __init__(self, mem_size, input_shape):
        self.mem_size = mem_size
        self.mem_cntr = 0
        
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.termination_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, obs, action, reward, obs_, done):
        idx = self.mem_cntr % self.mem_size
        
        self.state_memory[idx] = obs
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.new_state_memory[idx] = obs_
        self.termination_memory[idx] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_size, self.mem_cntr)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        dones = self.termination_memory[batch]

        return states, actions, rewards, new_states, dones
