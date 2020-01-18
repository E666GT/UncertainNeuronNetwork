import torch
from Paras import Parameters
from torch.utils.data import Dataset
import random

class Simple_DataSet(Dataset):
    def __init__(self):
        self.p=Parameters()
        self.Input_Data_List=[i for i in range(self.p.Len_Dataset)]
        self.Output_Data_list=[i*2+1+random.random()*0.1 for i in range(self.p.Len_Dataset)]

        pass
    def __len__(self):
        return self.p.Len_Dataset

    def __getitem__(self,index):
        x = self.Input_Data_List[index]
        y = self.Output_Data_list[index]
        sample={"x":x,"y":y}
        return sample