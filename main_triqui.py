from Tictactoe_class import TicTacToe
from agente_deterministico import AgenteDet
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as DQMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.env_checker import check_env
from stable_baselines import *
import gym
import pygame 
import math 
import os.path
import pickle
import argparse 

def test():
    agent = ACER.load('DQN_triqui_v1.2')
    render = True
    triqui = TicTacToe(agent,render,mode='train',epsilon=0)
    run = True
    done = False
    past_row = None
    past_col = None
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                m_x,m_y=click()
                human_row = math.floor(m_y/triqui.gap)
                human_col = math.floor(m_x/triqui.gap)
                curr_action = human_row*triqui.columns+human_col
                observation, reward, done, info=triqui.step(curr_action)

        triqui.render()
        if done:
            triqui.display_message('acabo')
            triqui.reset()
            observation = triqui.grid
            done = False
        
        

def click():
    m_x, m_y = pygame.mouse.get_pos()
    return m_x,m_y


def train():
    agent = AgenteDet()

    triqui = TicTacToe(base_agent=agent,render=False,mode='primero',epsilon=0)
    #model = DQN.load('DQN_triqui_v1.1',env=triqui,tensorboard_log="./triqui/",verbose=1,learning_rate=0.00003,prioritized_replay=True,batch_size=1024) #train_freq=10,target_network_update_freq=500
    model = ACER(MlpPolicy,env=triqui,verbose=1,tensorboard_log="./triqui/")
    model.learn(total_timesteps=200000,tb_log_name="Acer1.1")
    
    triqui = TicTacToe(base_agent=agent,render=False,mode='segundo',epsilon=0)
    model.env = triqui
    model.learn(total_timesteps=200000,tb_log_name="Acer1.2",reset_num_timesteps=False)

    # triqui = TicTacToe(base_agent=agent,render=False,mode='test',epsilon=0.4)
    # model.env = triqui
    # model.learn(total_timesteps=50000,tb_log_name="AgenteOpuestoContraAgente1.3")

    # triqui = TicTacToe(base_agent=agent,render=False,mode='test',epsilon=0.3)
    # model.env = triqui
    # model.learn(total_timesteps=70000,tb_log_name="AgenteOpuestoContraAgente1.4")

    # triqui = TicTacToe(base_agent=agent,render=False,mode='test',epsilon=0.2)
    # model.env = triqui
    # model.learn(total_timesteps=100000,tb_log_name="AgenteOpuestoContraAgente1.5")

    #triqui = TicTacToe(base_agent=agent,render=False,mode='test')
    #check_env(triqui)
    #model.learn(total_timesteps=50000,tb_log_name="ConO",reset_num_timesteps=False)
    model.save('ACER_triqui_v1')

    evaluation(triqui,model)



def evaluation(env,agent):
    obs = env.reset()

    n_simulations = 800
    winnings = 0
    matches = 1
    for i in range(n_simulations):
        action, _states = agent.predict(obs)
        print('action: ',action)
        obs, rewards, dones, info = env.step(action)
        print(env.grid)
        print('Reward: ',rewards)
        #  env.render()
        if dones == True:
            matches+=1
            if rewards==1:
                winnings+=rewards
            print('-------------------------------------')
            print('Fin del juego')
            print('-------------------------------------')
            obs = env.reset()
    print('Winning percentage: ',str(winnings/matches))

def main(mode='train'):
    if mode=='train':
        train()
    elif mode == 'test':
        test()
    elif mode == 'evaluate':
        agent = ACER.load('ACER_triqui_v1')
        opp_agent = AgenteDet()
        env = TicTacToe(base_agent=opp_agent,render=False,mode='segundo',epsilon=0)
        evaluation(env,agent)
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Choose the operation mode')
    parser.add_argument('mode', metavar='N', type=str,
                    help='operation mode, could be train or test')
    args = parser.parse_args()
    main(args.mode)