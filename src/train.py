from src.model import SimpleNeuralNetwork
from src.dataset import CocoDetectionDataset
from torch.ultis.data import DataLoader
import torch.nn as nn
if __name__ == '__main__':
    num_epoch = 50
    train_dataset = CocoDetectionDataset(root="/Users/Documents/DL-Project/OD_DA2_NN_v1/data/dataset/train",train=True)
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = 8,
        shuffle = True,
        num_worker = 0,
        drop_last = False
    )
    valid_dataset = CocoDetectionDataset(root="/Users/Documents/DL-Project/OD_DA2_NN_v1/data/dataset/train",train=True)
    valid_dataloader = DataLoader(
        dataset = valid_dataset,
        batch_size = 8,
        shuffle = True,
        num_worker = 0,
        drop_last = False
    )
    model = SimpleNeuralNetwork(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.001, momentum = 0.9)
    num_iter = len(train_dataloader)
    for i in range(num_epoch):
        model.train()
        for  num_iter,(images,label) in train_dataloader:
            outputs = model(images)
            loss_value = criterion(outputs,label)
            print("Epoch{}/{}. Iteration {}/{}. Loss{}".format(num_epoch+1,num_epoch,num_iter+1,num_iter,loss_value))
                        #BACKWARD

            optimizer.zero_grad()# trong pytorch mac dinh moi lan dungf backward mmoi lan n laij luu tru gradient 
            loss_value.backward()# tinh gradient
            optimizer.step()# cap nhat lai tham so