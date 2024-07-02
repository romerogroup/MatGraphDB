from torch.utils.data import Dataset

# Custom Dataset class
class NumpyDataset(Dataset):
    def __init__(self, X,y):
        self.X = X
        self.y = y
        self.n_samples=len(self.X)

    def __getitem__(self, index):
        features = self.X[index,:]
        label = self.y[index]
        return features, label
    
    def __len__(self):
        return self.n_samples