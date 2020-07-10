from make_env import make_env
import numpy as np

import maddpg_class
from maddpg_class import MADDPG

import ipdb

#Agent init_params needs to store agents in the proper order
if __name__=='__main__':
    agent_init_params=agent_init_params=[{'alpha':0.01, 'beta':0.01, 'input_dims':21, 'n_actions_physical':5,'n_actions_communication':10},
                                       {'alpha':0.01, 'beta':0.01, 'input_dims':21, 'n_actions_physical':5,'n_actions_communication':10}]
    ma=MADDPG(agent_init_params,env='simple_reference')
    
    #simple_init_params=[{'alpha':0.04, 'beta':0.03, 'input_dims':4, 'n_actions_physical':2,'n_actions_communication':0}]
    #ma=MADDPG(simple_init_params,env='simple')
    
    n_episodes=10000

    steps_per_episode=50
    for i in range(n_episodes):
        obs=ma.reset()
        done=[1]
        
        agent_mean_reward=[]
        step_count=0
        #Begin Episode
        while(sum(done)<ma.nagents and step_count<steps_per_episode):
            next_obs,rewards,done=ma.step(obs)
            agent_mean_reward.append(np.mean(rewards))

            ma.learn()
            obs=next_obs
            step_count+=1
            ma.env.render()
            ma.update_networks()
        

        print('Reward for Episode',i,'=',np.sum(agent_mean_reward))