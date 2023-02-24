import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import tensor
import random
from collections import deque

import numpy as np

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            device: device to put experiences when sampling
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.device = device
    
    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""

        for s,a,r,sp,d in zip(states, actions, rewards, next_states, dones):
            self.memory.append([s, a, r, sp, d ])    


    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        s, a, r, s_prime, done = zip(*experiences)
        
        states = tensor(np.array(s)).float().to(self.device)
        actions = tensor(np.array(a)).float().to(self.device)
        rewards = tensor(np.array(r)).float().to(self.device)
        next_states = tensor(np.array(s_prime)).float().to(self.device)
        dones = tensor(np.array(done)).float().to(self.device)
  
        return states, actions, rewards, next_states, dones


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Actor(nn.Module):
    def __init__(self, obs_size: int, act_size: int, scale: float, hidden_1: int, hidden_2: int):
        super().__init__()
        self.output_scale_factor = scale
        self.model = nn.Sequential(nn.Linear(obs_size, hidden_1), nn.ReLU(),
                                   nn.Linear(hidden_1, hidden_2), nn.ReLU(),
                                   nn.Linear(hidden_2,act_size), nn.Tanh())
        

    def forward(self, s: tensor) -> tensor:
        """
        Returns a tuple with deterministic continuous action to take
        """
        x = self.model(s)
        return x * self.output_scale_factor

    
class Critic(nn.Module):
    def __init__(self, obs_size: int, act_size: int, obs_hidden_1: int, act_hidden_1: int, hidden_2: int, hidden_3: int):
        super().__init__()
        self.obs_net   = nn.Sequential(nn.Linear(obs_size, obs_hidden_1), nn.ReLU())        
        self.a_net = nn.Sequential(nn.Linear(act_size, act_hidden_1), nn.ReLU())
        
        self.q_net = nn.Sequential(nn.Linear(obs_hidden_1+act_hidden_1, hidden_2), nn.ReLU(),
                                   nn.Linear(hidden_2, hidden_3), nn.ReLU(),
                                   nn.Linear(hidden_3, 1))        

    def forward(self, s: tensor, a: tensor) -> tensor:
        """
        Returns a tuple with deterministic continuous action to take
        """
        x_obs = self.obs_net(s)
        x_act = self.a_net(a)
        x = self.q_net(torch.cat([x_obs,x_act], dim=1))
        return x
    

class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class DDPGAgent():
    """DDPG Agent implementation"""

    def __init__(self, actor_state_size, actor_action_size, critic_state_size, critic_action_size, device, **hparam):
        """Initialize an Agent object.
        
        Params
        ======
            actor_state_size (int): dimension of each state
            actor_action_size (int): dimension of each action
            critic_state_size (int): dimension of each state
            critic_action_size (int): dimension of each action
            device: device to run model
            hparam: dictionary with hyper parameters
           
        """
        self.actor_state_size = actor_state_size
        self.actor_action_size = actor_action_size
        self.critic_state_size = critic_state_size
        self.critic_action_size = critic_action_size        
        self.hparam = hparam
        self.device = device
        self.train_steps = 0

        # noise 
        self.noise = OUNoise(actor_action_size, 100, sigma=hparam['SIGMA'], theta=hparam['THETA'])       

        # Actor
        self.actor_local = Actor(actor_state_size, actor_action_size, scale=hparam["OUT_SCALE"],
                                 hidden_1=hparam["ACTOR_HIDDEN_1"], hidden_2=hparam["ACTOR_HIDDEN_2"]).to(device)
        self.actor_target = Actor(actor_state_size, actor_action_size, scale=hparam["OUT_SCALE"],
                                  hidden_1=hparam["ACTOR_HIDDEN_1"], hidden_2=hparam["ACTOR_HIDDEN_2"]).to(device)        
        self.actor_target.load_state_dict( self.actor_local.state_dict())
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=hparam["ACTOR_LR"])

        # Critic
        self.critic_local = Critic(critic_state_size, critic_action_size,
                                   obs_hidden_1=hparam["CRITIC_OBS_HIDDEN_1"], act_hidden_1=hparam["CRITIC_ACT_HIDDEN_1"], 
                                   hidden_2=hparam["CRITIC_HIDDEN_2"], hidden_3=hparam["CRITIC_HIDDEN_3"]).to(device)
        self.critic_target = Critic(critic_state_size, critic_action_size,
                                    obs_hidden_1=hparam["CRITIC_OBS_HIDDEN_1"], act_hidden_1=hparam["CRITIC_ACT_HIDDEN_1"], 
                                    hidden_2=hparam["CRITIC_HIDDEN_2"], hidden_3=hparam["CRITIC_HIDDEN_3"]).to(device)
        self.critic_target.load_state_dict( self.critic_local.state_dict())
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=hparam["CRITIC_LR"])
        

        # Replay memory
        self.memory = ReplayBuffer(hparam["BUFFER_SIZE"], hparam["BATCH_SIZE"], device)

    
    def step(self, states, actions, rewards, next_states, dones):
        """
        Update the Agent with the next step from the environment
        """        
        # Save experience in replay memory
        self.memory.add(states, actions, rewards, next_states, dones)


    def act(self, states):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            states (array_like): current state           
        """
        states = torch.from_numpy(states).float().to(self.device)

        # set the NN to not train 
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states.reshape(1,-1)).cpu().detach().numpy() + self.noise.sample() # add noise to enforce exploration
        # set the NN again to train
        self.actor_local.train()
        
        return np.clip(actions, -1, 1)


    def learn(self, all_local_actions, all_target_next_actions, all_agent_experiences, rewards, dones):
        """Update value parameters using batch of experiences

        """
        gamma = self.hparam["GAMMA"]
        all_states, all_actions, _, all_next_states, _ = all_agent_experiences

        ## compute and minimize the critic loss
        target = rewards + gamma * self.critic_target(all_next_states, all_target_next_actions) * (1 - dones)
        critic_loss = F.smooth_l1_loss(self.critic_local(all_states,all_actions), target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute a minimize actor loss
        # all_local_actions is self.actor_local(single_states)
        actor_loss = -self.critic_local(all_states, all_local_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()        

        # ------------------- update target network ------------------- #
        self.soft_update(self.actor_local, self.actor_target, self.hparam["TAU"])
        self.soft_update(self.critic_local, self.critic_target, self.hparam["TAU"])
        self.train_steps += 1

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    

class MADDPGAgent():
    def __init__(self, state_size, action_size, num_agents, device, **hparam):
        self.agents = [ DDPGAgent(state_size, action_size, num_agents * state_size, num_agents * action_size, device, **hparam) for _ in range(num_agents)]
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.hparam = hparam

    def step(self, states, actions, rewards, next_states, dones):
        """
        Update the Agent with the next step from the environment
        """
        for i,agent in enumerate(self.agents):
            agent.step([states[i]], [actions[i]], [rewards[i]], [next_states[i]], [dones[i]])

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.hparam["UPDATE_EVERY"]
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            memory_lens = np.array( [len(agent.memory) for agent in self.agents] )
            if np.all( memory_lens > self.hparam["BATCH_SIZE"]):
                for _ in range(self.hparam["K"]):
                    self.learn()

        

    def act(self, all_states):
        """Returns actions for all agents given states as per current policy.
        
        Params
        ======
            states (array_like): current state per agent
        """
        all_actions = []
        for i,agent in enumerate(self.agents):
            actions = agent.act(all_states[i])
            all_actions.append(actions)
            
        return np.hstack(all_actions)
            
    def learn(self):
        """Update value parameters using batch of experiences

        """
        all_local_actions = []
        all_target_next_actions = []
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []
        
        for agent in self.agents:
            states, actions, rewards, next_states, dones = agent.memory.sample()
            
            local_actions = agent.actor_local(states)
            target_next_actions = agent.actor_target(next_states)            
            all_local_actions.append(local_actions)
            all_target_next_actions.append(target_next_actions)
            
            all_states.append(states)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_next_states.append(next_states)
            all_dones.append(dones)
            

        all_experiences = torch.cat(all_states, dim=1), torch.cat(all_actions, dim=1), torch.cat(all_rewards, dim=1), torch.cat(all_next_states, dim=1), torch.cat(all_dones, dim=1)
        # feed each agent with all experiences and all agent actor actions
        for i,agent in enumerate(self.agents):
            # detach all actions except the current one, so only that is modified during learning
            all_local_actions_detached = [actions if j == i else actions.detach() for j,actions in enumerate(all_actions)]
            agent.learn(torch.cat(all_local_actions_detached, dim=1) , torch.cat(all_target_next_actions, dim=1), all_experiences, all_rewards[i], all_dones[i])