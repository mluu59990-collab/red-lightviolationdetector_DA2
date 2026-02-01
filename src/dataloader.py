from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from src.dataset import CocoDetectionDataset
if __name__ == '__main__':
    training_data = CocoDetectionDataset(root='/Users/Documents/DL-Project/OD_DA2_NN_v1/data/dataset',
                                         transform=ToTensor(),
                                         train=True)
    
    valid_data = CocoDetectionDataset(root='/Users/Documents/DL-Project/OD_DA2_NN_v1/data/dataset',
                                     transform=ToTensor(),
                                     train=False)
    
    def collate_fn(batch):
        return tuple(zip(*batch))
    train_dataloader = DataLoader(
        dataset = training_data,
        batch_size = 16,
        num_workers = 0,# mays co bao nhieu nhan thi su dung
        drop_last = False,# Xoa nhung anh ma batch size chia con thua
        shuffle = True, #Shuffle lai sau 1 epoch
        collate_fn = collate_fn
    )
    test_dataloader = DataLoader(
        dataset = valid_data,
        batch_size = 4,
        num_workers = 4,
        drop_last = False,
        shuffle = False,
        collate_fn = collate_fn
    )
    for imgs, targets in train_dataloader:
        print(type(imgs))        # tuple
        print(len(imgs))         # batch size

        print(imgs[0].shape)     # tensor ảnh đầu tiên
        print(targets[0])        # annotation dict ảnh đầu tiên
        break
