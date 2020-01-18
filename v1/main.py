import numpy as np
import os
import env


SIZE=110

MAX_ENERGY_STORE_EACH=10000 #每个神经元最大存储100能量
MAX_ENERGY_TRANS_EACH=100 #每个连接，最大传递100能量

class UncertainNetwork(object):
    def __init__(self,size):
        self.size=size

        self.InitWeights()
        self.EnergyStorage=np.random.random([size]) #每一个神经元中，有一个engergy，engergy通过connect与相邻神经元进行weights加权传递
        self.EngergySpreadReceiveBuffer=np.zeros([self.size]) #全局传递能量一次，先存入buffer，再从buffer加载入各自的能量
        self.EngergySpreadSendBuffer=np.zeros([self.size]) #全局传递能量一次，先存入buffer，再从buffer加载入各自的能量
        self.MAX_ENERGY_STORE_EACH=MAX_ENERGY_STORE_EACH

        #功能分化
        self.ChargeIDs=np.array(range(1,10))#充电的神经元
        self.RewardIDs=np.array(range(20,25))

        self.CartPole_Eye_IDs=np.array([100,101,102,103,104,105])
        self.CartPole_Control_IDs=np.array([106])
        self.CartPole_Excution_Energy_Store=0
    def EnergyClear(self,IDs):
        self.EnergyStorage[IDs]=0
    def EnergyInput(self,ReceiverID,Engergy):
        self.EnergyStorage[ReceiverID]+=Engergy
        self.EnergyStorage=np.clip(self.EnergyStorage,0,self.MAX_ENERGY_STORE_EACH)
    def StepGlobalEnergySpread(self):
        self.ClearEngergySpreadReceiveBuffer()
        self.ClearEngergySpreadSendBuffer()
        self.StepEnergyCharge()
        for row in range(self.Weights.shape[0]):
            SenderID=row
            # print("row",row)
            energy = self.EnergyStorage[row]
            for col in range(self.Weights.shape[1]):
                ReceiverID=col
                weight=self.Weights[row,col]
                if(SenderID==ReceiverID):#不能给自己传递
                    continue
                self.P2PEngergySpread(SenderID=SenderID,ReceiverID=ReceiverID)
        self.UpdateEnergyStorage()
        self.ClearEngergySpreadReceiveBuffer()
        self.ClearEngergySpreadSendBuffer()

    def StepEnergyCharge(self):
        for id in self.ChargeIDs:
            self.EnergyStorage[id]=MAX_ENERGY_STORE_EACH
    def P2PEngergySpread(self,SenderID,ReceiverID):
        weight=self.Weights[SenderID,ReceiverID]
        input_energy=self.EnergyStorage[SenderID]/np.sum(self.Weights[SenderID,:])*0.6*np.random.random()
        # print("EnergyStorage",self.EnergyStorage[SenderID])
        # print("input_energy",input_energy)
        # decline_rate=np.random.random() #能量加权均分
        decline_rate=1 #能量加权均分
        # print("ID's sum weights:",SenderID,"  ",np.sum(self.Weights[SenderID,:]))
        ReceivedEnergy=input_energy*weight*decline_rate
        self.EngergySpreadReceiveBuffer[ReceiverID]+=ReceivedEnergy
        if(SenderID in self.CartPole_Control_IDs):#执行器 不发送能量
            self.EngergySpreadSendBuffer[SenderID]+=0
        else:
            self.EngergySpreadSendBuffer[SenderID]+=input_energy

        # if(SenderID)
        # if (SenderID == 106):
        #     print("sd=106, energy=", input_energy)
        #     print("sd=106, energy=", input_energy)
        # if (ReceiverID == 106):
        #     print("rd=106, energy=", ReceivedEnergy)
        #     print("rd=106, energy=", ReceivedEnergy)

        #weights updating
        weight=weight
        grow_energy_thresh=MAX_ENERGY_STORE_EACH/self.size*0.05 #!!!
        # print("input energy",input_energy)
        # print("grow_energy_thresh",grow_energy_thresh)
        weight_grow_rate=np.clip(((input_energy-grow_energy_thresh)/MAX_ENERGY_STORE_EACH),-0.01,0.01)

        # print("weight_grow_rate",weight_grow_rate)
        new_weight=np.clip(weight*(1+weight_grow_rate),0.01,1)
        # if (SenderID == 106):
        #     print("weight_grow_rate=", weight_grow_rate)
            # print("weight_grow_ratey=", input_energy)
        self.Weights[SenderID,ReceiverID]=new_weight

        # self.EngergySpreadReceiveBuffer[SenderID]-=
        # self.EngergySpreadSendBuffer[SenderID]
        # print("input_energy",input_energy)
        # print("weight",weight)
        # print("ReceivedEnergy",ReceivedEnergy)
        return True
    def UpdateEnergyStorage(self):
        #developing 一次性全部传递 还是 按时间一点一点传递呢! !! 应该是按时间一点一点传递
        # self.EnergyStorage=self.EnergyStorage+self.EngergySpreadReceiveBuffer #将buffer中传递信息，更新到全局能量storage中
        # 将buffer中传递信息，更新到全局能量storage中
        self.EnergyStorage=self.EnergyStorage+self.EngergySpreadReceiveBuffer-self.EngergySpreadSendBuffer
        self.EnergyStorage=np.clip(self.EnergyStorage,0,MAX_ENERGY_STORE_EACH)
        # print("EngergySpreadReceiveBuffer",self.EngergySpreadReceiveBuffer[100:105])
        # print("EngergySpreadSendBuffer",self.EngergySpreadSendBuffer[100:105])
        # print("UpdateEnergyStorage",self.EnergyStorage[100:105])

    def SaveWeights(self,filename="weights"):
        path="weights"
        if(path not in os.listdir()):
            os.mkdir(path)
        np.save(path+"\\"+filename+".npy",self.Weights)
        print("weights已保存")
    def LoadWeights(self,filename="weights"):
        path="weights"
        self.Weights=np.load(path+"\\"+filename+".npy")
        print("weights加载成功")
    def InitWeights(self):
        self.DefultWeights = np.random.random([self.size, self.size])
        self.Weights=self.DefultWeights
    def ClearEngergySpreadReceiveBuffer(self):
        self.EngergySpreadReceiveBuffer = np.zeros([self.size])
    def ClearEngergySpreadSendBuffer(self):
        self.EngergySpreadSendBuffer = np.zeros([self.size])


if __name__=="__main__":
    #init network
    network=UncertainNetwork(SIZE)
    #init player trainer
    player=env.player()
    # network.StepGlobalEnergySpread()

    # simu_steps=10
    # for i in range(simu_steps):
    #     network.StepGlobalEnergySpread()
    #     print(network.EnergyStorage[1])

    #play cartpole
    player.play_cartpole(network=network,MaxGameLoops=10000,ControlSpreads=1)




