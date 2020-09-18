from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .transforms import tensor_transform
import os


class Dataset(Dataset):
    def __init__(self, dataset_path, transform=tensor_transform):
        import csv
        from os import path
        self.data = []
        self.path_data = []  # optional, can be helpful to store
        with open(path.join(dataset_path, 'labels.csv'), newline='') as f:
            reader = csv.reader(f)
            for fname, label in reader:
                # if label in LABEL_NAMES:
                # fname = "/Users/juanhuerta/work/work_projects/Classification_started_code/Collect_data/images" + fname
                image = Image.open(fname)
                label_id = int(label)
                self.data.append((image, label_id))
                self.path_data.append((fname, label_id))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, lbl = self.data[idx]
        return self.transform(img), lbl


def load_data(dataset_path, transforms=tensor_transform, num_workers=0, batch_size=32, random_seed=40):
    '''
    this data loading proccedure assumes dataset/train/ dataset/val/ folders
    also assumes transform dictionary with train and val
    '''
    dataset_train_path = dataset_path + "/train"
    dataset_train = Dataset(dataset_train_path, transform=transforms['train'])

    dataset_val_path = dataset_path + "/val"
    dataset_val = Dataset(dataset_val_path, transform=transforms['valid'])

    print("Size of train dataset: ", len(dataset_train))
    print("Size of val dataset: ", len(dataset_val))

    dataloaders = {
        'train': DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True),
        'val': DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=True)
    }
    return dataloaders


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()


def overrides(interface_class):
    '''
    overwrite helper function. this checks that we did in fact properly overwrite a method
    :param interface_class: class of method we are overwritting
    :return: assertion
    '''

    def overrider(method):
        assert (method.__name__ in dir(interface_class))
        return method

    return overrider


def newest_model(path='Model/saved_models'):
    '''
    This function finds newest file in dir
    we use this to find most recent model to default to
    :param path: saved_models path directory
    :return: str path with most recent file
    '''
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)
