#There's a different critic for every agent, but it's centralized because it considers the actions of every agent

#The critic takes in a state vector as the observations of ALL agents.
#When updating the critic weights, we need to know the other agent actions at next states. At present, we assume that the agents choose according to a target critic.

#The target critic of all agents should be known by each agent.

#Sampling from the gumbel softmax distribution allows us to use backprop

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain

from utils import OUActionNoise,ReplayBuffer


class CriticNetwork(nn.Module):
    def __init__(self,beta, input_dims, fc1_dims, fc2_dims, name,
                 chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        
        #input dims consist of all observations and all actions

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_ddpg')
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])

        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1=nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        #f2 = 0.002
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        #self.fc2.weight.data.uniform_(-f2, f2)
        #self.fc2.bias.data.uniform_(-f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        torch.nn.init.uniform_(self.q.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.q.bias.data, -f3, f3)
        #self.q.weight.data.uniform_(-f3, f3)
        #self.q.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, states, actions,actions_need_processing=True):
        states=list(chain.from_iterable(states)) #To concatenate this into a single list
        states=torch.tensor(states,dtype=torch.float32).to(self.device)

        if actions_need_processing:
            actions=list(chain.from_iterable(actions)) #To concatenate this into a single list
            actions=torch.tensor(actions,dtype=torch.float32).to(self.device)

        state_action_input=torch.cat((states,actions)).to(self.device)

        self.eval()
        state_action_value = self.fc1(state_action_input)
        state_action_value = self.bn1(state_action_value)
        state_action_value = F.relu(state_action_value)
        state_action_value = self.fc2(state_action_value)
        state_action_value = self.bn2(state_action_value)
        state_action_value = self.q(state_action_value)

        return state_action_value


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))    



class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name,discrete=True,
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_ddpg')
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])

        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1=nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        #f2 = 0.002
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        #self.fc2.weight.data.uniform_(-f2, f2)
        #self.fc2.bias.data.uniform_(-f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.fc3=nn.Linear(self.fc2_dims,n_actions)
        f3 = 1./np.sqrt(self.fc3.weight.data.size()[0])
        #f2 = 0.002
        torch.nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.fc3.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        #Communication actions are separate from physical actions.

    def forward(self,observations):
        observations=torch.tensor(observations,dtype=torch.float32).to(self.device)
        x = self.fc1(observations)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x=torch.tanh(self.fc3(x))
        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))