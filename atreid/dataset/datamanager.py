from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, cid, mid, camid = self.dataset[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, pid, cid, mid, camid, index


if __name__ == '__main__':
    pass
