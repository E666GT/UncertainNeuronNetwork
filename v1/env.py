import gym
import numpy as np
class player(object):
  def __init__(self):
    pass
  def play_cartpole(self,network,ControlSpreads,MaxGameLoops=100):
    weight_filename="CartPole"
    CartPole_Control_IDs = network.CartPole_Control_IDs
    CartPole_Eye_IDs = network.CartPole_Eye_IDs
    Reward_IDs = network.RewardIDs
    def get_action(network=network):

      energy=np.mean(network.EnergyStorage[CartPole_Control_IDs])
      print("action energy",energy)
      thresh_energy=network.MAX_ENERGY_STORE_EACH/5
      if(energy>thresh_energy):
        control =1
      else:
        control = 0
      network.EnergyClear(CartPole_Control_IDs)
      return control
    def InputObservation(observation,network=network):
      observation_id=0
      CartPosition=observation[0]#-4.8 4.8
      CartVel=observation[1] # -inf +inf
      Angle=observation[2] #-24/180 24/180
      TipVel=observation[3]#-inf +inf
      network.EnergyInput(CartPole_Eye_IDs[0],Engergy=(CartPosition+4.8)/9.6*network.MAX_ENERGY_STORE_EACH)
      network.EnergyInput(CartPole_Eye_IDs[1],Engergy=(CartVel+5)/10*network.MAX_ENERGY_STORE_EACH)
      network.EnergyInput(CartPole_Eye_IDs[2],Engergy=(Angle+24/180)/(48/180)*network.MAX_ENERGY_STORE_EACH)
      network.EnergyInput(CartPole_Eye_IDs[3],Engergy=(TipVel+5)/10*network.MAX_ENERGY_STORE_EACH)


      #
      # for id in CartPole_Eye_IDs:
      #   # network.EnergyStorage[id]=
      #   # print("observation",observation)
      #   energy=network.MAX_ENERGY_STORE_EACH*observation[observation_id]
      #   network.EnergyInput(id,Engergy=energy)
      #   observation_id+=1
    def InputReward(reward,scores_reward,network=network):
      network.EnergyInput(Reward_IDs[0], Engergy=network.MAX_ENERGY_STORE_EACH *reward)
      network.EnergyInput(Reward_IDs[1], Engergy=network.MAX_ENERGY_STORE_EACH *reward)
      network.EnergyInput(Reward_IDs[2], Engergy=network.MAX_ENERGY_STORE_EACH *scores_reward)
      network.EnergyInput(Reward_IDs[3], Engergy=network.MAX_ENERGY_STORE_EACH *pow(scores_reward,2))
      network.EnergyInput(Reward_IDs[4], Engergy=network.MAX_ENERGY_STORE_EACH *pow(scores_reward,5))

    try:
      network.LoadWeights(filename=weight_filename)
      # print("weights加载成功")
    except:
      network.InitWeights()
    env = gym.make("CartPole-v1")
    observation = env.reset()
    # for _ in range(1000):
    game_loops=0
    scores=0
    av_scores=0
    total_scores=0
    max_scores=1
    while(1):
      max_scores=max(scores,max_scores)
      scores+=1
      total_scores+=1
      # print(_)
      env.render()
      action = get_action()
      # print('action',action)
      observation, reward, done, info = env.step(action)
      InputObservation(observation)
      scores_reward=scores/max_scores
      InputReward(reward,scores_reward)
      i=0
      while(i<ControlSpreads):
        network.StepGlobalEnergySpread()
        # print(network.EnergyStorage[CartPole_Eye_IDs],network.EnergyStorage[CartPole_Control_IDs])
        # print("100->50 connect:",network.Weights[100,50])
        # print("1->100 connect:",network.Weights[1,100])
        # print("100:105 storage",network.EnergyStorage[100:105])
        # print("1:10 storage",network.EnergyStorage[1:10])
        # print("100:105，：10 weights",network.Weights[100:105,:10])
        # print("1:10 weights ",network.Weights[1:10,100:105])
        # print("90:95,106 weights",network.Weights[90:95,106])
        i+=1
      # print(observation)
      if done:

        game_loops+=1
        av_scores=total_scores/game_loops
        observation = env.reset()
        network.SaveWeights(filename="CartPole")
        print("上一局Score=",scores,",正在玩第",game_loops,"局","av_score=",av_scores)
        scores=0
        if(game_loops>MaxGameLoops):
          break
    env.close()