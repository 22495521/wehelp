import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyDataset(Dataset):
    def __init__(self):
        self.x = torch.tensor([[1.], [2.], [3.], [4.], [5.], [6.]])
        self.y = torch.tensor([10., 20., 30., 40., 50., 60.])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

ds = MyDataset()

loader = DataLoader(ds, batch_size=64,shuffle=True)

for xb, yb in loader:
    print(77777,xb, yb)