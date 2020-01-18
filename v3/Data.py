import torch
# from GlobalDB import DB
from torch.utils.data import Dataset
import random
import numpy as np

np.random.seed(2019)

class Simple_DataSet(Dataset):
    def __init__(self,db,type):
        # self.p=DB()
        self.db=db
        self.type=type
        if(type=="train"):
            self.Input_Data_List=[i for i in range(self.db.Len_Dataset_Train)]
            self.Output_Data_List=[self.get_output(i) for i in range(self.db.Len_Dataset_Train)]
        elif(type=="test"):
            self.Input_Data_List=np.random.randint(1,180000,[db.Len_Dataset_Test])
            self.Output_Data_List=[self.get_output(i) for i in self.Input_Data_List]
        pass
    def get_output(self,i):
        return i*2+1+random.random()*0.1
    @property
    def train(self):
        return self.type=="train"
    @property
    def test(self):
        return self.type=="test"

    def __len__(self):
        if self.train:
            return self.db.Len_Dataset_Train
        if self.test:
            return self.db.Len_Dataset_Test

    def __getitem__(self,index):
        x = self.Input_Data_List[index]
        y = self.Output_Data_List[index]
        sample={"x":x,"y":y}
        return sample