import torch
import torch.nn as nn


class simple_net(nn.Module):
    def __init__(self,db):
        super(simple_net, self).__init__()
        self.linear1=nn.Linear(in_features=db.batch_size,out_features=12).cuda()
        self.linear2=nn.Linear(in_features=12,out_features=db.batch_size).cuda()
        pass
    def forward(self,x):
        x=x.float()
        x=x.cuda()
        x=self.linear1(x)
        x=self.linear2(x)
        return x
