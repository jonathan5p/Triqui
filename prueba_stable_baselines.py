import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import argparse

def prueba_baselines():
    env = gym.make('CartPole-v1')
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    # env = DummyVecEnv([lambda: env])

    model = PPO2(MlpPolicy, env, verbose=0)
    model.learn(total_timesteps=10)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones == True:
            obs = env.reset()
    pass

def main():
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Choose the operation mode')
    parser.add_argument('mode', metavar='N', type=str,
                    help='operation mode, could be train or test')

    args = parser.parse_args()
    print(type(args.mode))
    #main()