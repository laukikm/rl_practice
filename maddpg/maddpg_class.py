import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from itertools import chain
from agent import Agent
from utils import ReplayBuffer

from make_env import make_env
from copy import deepcopy
#The order in which you initialize the agents may not match that of the multi-agent repo
#At the moment, env.reset() returns 21 observations for both agents, but env.step() return only 16 and 11 respectively

#All the agents achieve their objectives at different instants

class MADDPG:
    def __init__(self, agent_init_params,batch_size=1024,replay_buffer_capacity=100000,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False,env='simple_reference'):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = len(agent_init_params)
        self.agents=[]

        n_actions_total=np.sum([agent['n_actions_physical']+agent['n_actions_communication'] for agent in agent_init_params])

        n_observations_total=np.sum([agent['input_dims'] for agent in agent_init_params])

        for i in range(self.nagents):

            current_agent=Agent(id_number=i,**agent_init_params[i])
            current_agent.initialize_critic(n_actions_total+n_observations_total)
            self.agents.append(current_agent)

        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr

        '''
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0
        '''
        
        self.replay_buffer=ReplayBuffer(replay_buffer_capacity)
        self.batch_size=batch_size

        self.env=make_env(env)

        self.loss=nn.MSELoss()

    def reset(self):
        return self.env.reset()

    def step(self,observations): #You have all the agents and policies. Once you have observations, you can just calculate actions to step

        #'observations' is a list of np arrays containing the observations of each agent.

        #'actions' should be a list of np arrays containing the actions of each agent. There's one np array per agent
        actions=[]


        for i,observation in enumerate(observations):
            current_action=self.agents[i].Action(observation)
            actions.append(current_action)

        

        new_states,rewards,terminals,_= self.env.step(actions)

        self.replay_buffer.store_transition(observations,actions,rewards,new_states,terminals)
        return new_states,rewards,terminals

    def update_networks(self,hard=False): #Carry out soft updates
        for agent in self.agents:
            agent.update_network_parameters(self.tau)

    def next_actions(self,next_states):
        next_step_actions=[]

        for i,agent in enumerate(self.agents):
            action=agent.Action(next_states[i],target=True)
            next_step_actions.append(action)

        return next_step_actions

    def learn(self):
        if self.replay_buffer.mem_cntr<self.batch_size: return

        states_batch,actions_batch,rewards_batch,next_states_batch,terminal_batch=self.replay_buffer.sample_buffer(self.batch_size)

        for agent in self.agents:
            #Critic Update
            self.update_critic(agent,states_batch,actions_batch,rewards_batch,next_states_batch,terminal_batch)
            self.update_actor(agent,states_batch,actions_batch,rewards_batch,next_states_batch,terminal_batch)



    def update_critic(self,agent,states_batch,actions_batch,rewards_batch,next_states_batch,terminal_batch):
        critic_losses=[]
        for i in range(self.batch_size):
            current_states=states_batch[i]
            current_actions=actions_batch[i]
            current_rewards=rewards_batch[i]
            next_states=next_states_batch[i]
            next_step_actions=[]

            for j,next_agent in enumerate(self.agents):
                action=next_agent.Action(next_states[j],target=True)
                next_step_actions.append(action)

            agent.critic.eval()                        
            Q=agent.critic.forward(current_states,current_actions).to(agent.critic.device)
            target=current_rewards[agent.id]+self.gamma*agent.critic.forward(next_states,next_step_actions).to(agent.critic.device).detach()
            
            loss=self.loss(Q,target)
            critic_losses.append(loss)

        agent.critic.train()
        critic_losses=torch.stack(critic_losses,0)
        mean_critic_loss=torch.mean(critic_losses).to(agent.critic.device)

        agent.critic.optimizer.zero_grad()
        mean_critic_loss.backward()

        nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic.optimizer.step()

    def update_actor(self,agent,states_batch,actions_batch,rewards_batch,next_states_batch,terminal_batch):
        Q_values=[]
        for i in range(self.batch_size):
            current_states=states_batch[i]
            current_actions=actions_batch[i]
            current_rewards=rewards_batch[i]

            agent.actor.eval()
            agent_action=agent.actor.forward(current_states[agent.id])#This is a tensor, not discretized. We could use gumbel softmax to approximate the discretization

            physical_actions=agent_action[0:agent.n_actions_physical]
            comm_actions=F.gumbel_softmax(agent_action[agent.n_actions_physical:],hard=True)

            agent_action=torch.cat((physical_actions,comm_actions))


            actions_for_critic=deepcopy(current_actions)

            for i in range(self.nagents):
                if(i==agent.id):
                    actions_for_critic[agent.id]=agent_action
                else:actions_for_critic[i]=torch.tensor(actions_for_critic[i],dtype=torch.float32).to(agent.critic.device)

            actions_for_critic=list(chain.from_iterable(actions_for_critic))
            actions_for_critic=torch.stack(actions_for_critic)

            Q= -agent.critic.forward(current_states,actions_for_critic,actions_need_processing=False)
            Q_values.append(Q)

        Q_values=torch.stack(Q_values,0)

        mean_Q=torch.mean(Q_values)
            
        agent.actor.optimizer.zero_grad()
        agent.actor.train()

        mean_Q.backward()

        nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
        agent.actor.optimizer.step()
