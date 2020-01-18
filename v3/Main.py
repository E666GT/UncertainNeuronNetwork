import Net
import LossFuncs
import torch
from multiprocessing import Process, freeze_support
from GlobalDB import DB
from Data import Simple_DataSet
import random
import numpy as np

random.seed(2019)
np.random.seed(2019)
torch.manual_seed(2019)

def Train(db):
    pass

if __name__ == "__main__":
    freeze_support()
    db = DB()
    # db._run_train()
    db._run_test()