import numpy as np
import torch
import torch.nn as nn
from networks import ActorNetwork,CriticNetwork
from utils import ReplayBuffer,OUActionNoise

class Agent:
    def __init__(self,id_number, alpha, beta, input_dims, n_actions_physical=5,n_actions_communication=5,gamma=0.99,tau=0.1,
                  max_size=1000, layer1_size=64,
                 layer2_size=64, batch_size=64):
        self.id=id_number
        self.input_dims=input_dims
        self.beta=beta

        self.n_actions_physical=n_actions_physical
        self.n_actions_communication=n_actions_communication
        self.n_actions=n_actions_physical+n_actions_communication

        self.gamma = gamma
        self.tau = tau #Soft update coefficient for the target networks

        self.memory = ReplayBuffer(max_size)
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, n_actions=self.n_actions,
                                  name='Actor')
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, n_actions=self.n_actions,
                                         name='TargetActor')
        
        self.noise = OUActionNoise(mu=np.zeros(self.n_actions))

    def initialize_critic(self,critic_input_dims,layer1_size=64,
                 layer2_size=64):

        #critic_input_dims is the total of all observations and actions of all agents

        self.critic = CriticNetwork(self.beta, critic_input_dims, layer1_size,
                                    layer2_size,
                                    name='Critic')

        self.target_critic = CriticNetwork(self.beta, critic_input_dims, layer1_size,
                                           layer2_size, 
                                           name='TargetCritic')


    def discretize_tensor(self,mu_prime_physical):
        if(self.n_actions_communication==0):return mu_prime_physical.cpu().detach().numpy()

        mu_physical=mu_prime_physical[0:self.n_actions_physical].to(self.actor.device)
        mu_comm=mu_prime_physical[self.n_actions_physical:].to(self.actor.device)

        index=torch.argmax(mu_comm)
        mu_comm=torch.tensor([0]*self.n_actions_communication,dtype=torch.float32).to(self.actor.device)
        mu_comm[index]=1

        mu_net=torch.cat((mu_physical,mu_comm))
        return mu_net.cpu().detach().numpy()

    def Action(self,observation,target=False):
        self.actor.eval()
        if target:
            mu_physical=self.target_actor.forward(observation).to(self.actor.device)    
        else:
            mu_physical=self.actor.forward(observation).to(self.actor.device)
    
        mu_prime_physical=mu_physical+torch.tensor(self.noise(),dtype=torch.float).to(self.actor.device)
        self.actor.train()
        return self.discretize_tensor(mu_prime_physical) #This is a discretized numpy array

    #The learning can happen in a separate MADDPG class, as the agents need information from all other agents too.

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)
