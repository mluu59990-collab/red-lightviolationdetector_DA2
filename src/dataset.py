import torch
from torch.utils.data import Dataset
import json
import os
import cv2
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import Resize, ToTensor, Compose
from PIL import Image
class CocoDetectionDataset(Dataset):
    def __init__(self, root, split="train", transform=None,train=True):
        self.root = root
        self.transform = transform
        img_dir = os.path.join(root, split, "img")
        anno_file = os.path.join(root, split, "anno", "_annotations.coco.json")

        with open(anno_file, "r") as f:
            coco = json.load(f)

        # id → filename
        self.images = {img["id"]: img for img in coco["images"]}

        # image_id → annotations
        self.annotations = {}
        for ann in coco["annotations"]:
            self.annotations.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = list(self.images.keys())
        self.img_dir = img_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]

        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        boxes = []
        label = []
        for anno in self.annotations.get(image_id,[]):
            x,y,w,h = anno["bbox"]
            boxes.append([x,y,x+w,y+h])
            label.append(anno["category_id"])
        target = {
            "boxes":torch.tensor(boxes,dtype = torch.float32),
            "label":torch.tensor(label,dtype = torch.int64),
            "image_id":torch.tensor([image_id])
        }
        if self.transform:
            img = self.transform(img)

        return img, target

if __name__ == '__main__':
    transform = Compose([
        Resize((200,200)),
        ToTensor()
    ])
    root = "/Users/Documents/DL-Project/OD_DA2_NN_v1/data/dataset"
    Dataset = CocoDetectionDataset(root=root,train=True,transform=transform)
    img,label = Dataset.__getitem__(650)
    pil_img = to_pil_image(img)
    pil_img.show()
    print(label)
