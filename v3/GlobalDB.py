import Net
from Data import Simple_DataSet
import torch
import os
import numpy as np
from matplotlib import pyplot as plt
class DB(object):
    def __init__(self):
        self.lr=1e-10
        self.MaxEpoches=200
        self.batch_size=12
        self.Len_Dataset_Train=10000
        self.Len_Dataset_Test=2000
        self.Trained_dir="Trained"
        self.Trained_Model_savepath=os.path.join(self.Trained_dir,"Trained.pt")

        self.Train_DataSet = Simple_DataSet(self,"train")
        self.Train_DataSetLoader = torch.utils.data.DataLoader(self.Train_DataSet, batch_size=self.batch_size,
                                                    shuffle=True, num_workers=0, drop_last=True)


        self.Test_DataSet = Simple_DataSet(self,"test")
        self.Test_DataSetLoader = torch.utils.data.DataLoader(self.Test_DataSet, batch_size=self.batch_size,
                                                    shuffle=False, num_workers=0, drop_last=True)



        self.model = Net.simple_net(self).cuda()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def _train_epoch(self):
        self.model.train()
        av_loss=0.
        for i,sample in enumerate(self.Train_DataSetLoader):
            ins=sample["x"]
            labels=torch.tensor(sample["y"]).cuda().type(torch.float)
            output= self.model(ins)
            loss=self.criterion(output,labels)
            # print("loss",loss.clone().detach().mean())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            av_loss+=loss.sum()
        av_loss = av_loss.clone().detach() / len(self.Train_DataSetLoader)
        return av_loss

    def _run_train(self):
        min_av_loss=1e10
        for epoch in range(self.MaxEpoches):
            # print(epoch)
            print("epoch,", epoch)

            av_loss = self._train_epoch()
            if av_loss<min_av_loss:
                min_av_loss=av_loss
                self.save()
                print("save model with av loss:",min_av_loss)
            if (epoch) % 1 == 0:
                print('Epoch[{}/{}], av_loss: {:.6f}'.format(epoch, self.MaxEpoches, av_loss))

    def _run_test(self):
        self.load()
        self.model.eval()
        av_loss=0.
        out_list=np.array([])
        for i,sample in enumerate(self.Test_DataSetLoader):
            ins=sample["x"].float()
            labels=torch.Tensor(sample["y"].float())
            out=self.model(ins).cpu()

            out_list=np.append(out_list,out.detach().numpy())
            av_loss+=(out-labels).sum()
            av_loss_0=(av_loss/i)

        fig, ax = plt.subplots()
        ax.plot(self.Test_DataSet.Input_Data_List[:],self.Test_DataSet.Output_Data_List[:])
        ax.plot(self.Test_DataSet.Input_Data_List[:len(out_list)],out_list)
        plt.show()

    def save(self):
        torch.save(self.model,self.Trained_Model_savepath)
    def load(self):
        self.model=torch.load(self.Trained_Model_savepath)