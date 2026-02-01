import torch
import torch.nn as nn

#Toi thieu phai dinh nghia 2 ham "init"&"forward"
#init : khai bao dung nhung layer nao
#forward: thu tu ket noi cua cac layer va cach truyen data
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, num_classes = 10 ):
        super().__init__() #Goij den ham khoi tao cua lop cha
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features = 3*32*32,out_features = 512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features = 512,out_features = 1024),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features = 1024,out_features = 256),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_features = 256,out_features = num_classes),
        )
    def forward(self,x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
if __name__ == '__main__':
    model = SimpleNeuralNetwork()
    sample_data = torch.rand(1,3,32,32)
    kq = model.forward(sample_data)
    print(kq.shape)

