import glob
from PIL import Image

import torchvision.transforms as transforms

from torch.utils.data import Dataset


class FacesDataset(Dataset):
    def __init__(self, root_dir, extention, transform=None):
        self.root_dir = root_dir
        self.image_files = glob.glob(f"{root_dir}/*.{extention}")
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx])
        if self.transform:
            image = self.transform(image)
        return image
