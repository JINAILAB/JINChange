from torch.utils.data import Dataset, DataLoader
import cv2

class CustomDataset(Dataset):
    def __init__(self, img_paths1, img_paths2, labels, transforms=None):
        self.img_paths1 = img_paths1
        self.img_paths2 = img_paths2
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        img_path1 = self.img_paths1[index]
        img_path2 = self.img_paths2[index]
        image1 = cv2.imread(img_path1)
        image2 = cv2.imread(img_path2)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            image1 = self.transforms(image=image1)
            image2 = self.transforms(image=image2)
        
        if self.labels is not None:
            label = self.labels[index]
            return image1, image2, label
        else:
            return image1, image2
    
    def __len__(self):
        return len(self.img_paths1)