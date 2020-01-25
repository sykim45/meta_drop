import torch
import torchvision
from torch.utils.data import DataLoader, Sampler
import numpy as np



def sampling_labels(trainset, num_samples):
    train_data = trainset.data
    train_label = trainset.targets
    uniq_labs = set(train_label)

    chunks = []
    for i in uniq_labs:
        indices = np.argwhere(np.array(train_label)==i).reshape(-1)
        data = train_data[indices]
        label = i
        chunks.append((indices,label))

    pos_indices = []
    neg_indices = []
    for chunk in chunks:
        ixlist = chunk[0].tolist()
        np.random.shuffle(ixlist)

        pos_indices.extend(ixlist[:num_samples])
        neg_indices.extend(ixlist[num_samples:])
    return torch.tensor(pos_indices), torch.tensor(neg_indices)



class LabelSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

