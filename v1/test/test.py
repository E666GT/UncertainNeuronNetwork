import gym
import time
import random
env = gym.make('CartPole-v0')
for i_episode in range(200):
    observation = env.reset()
    for t in range(100):
        time.sleep(0.1)
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()