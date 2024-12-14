from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        d = {}

        for key in self.inputs[index].keys():
            d[key] = self.inputs[index][key][0]

        return d
