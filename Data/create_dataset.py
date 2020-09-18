'''
given a folder "Data/images" & csv file with image per row we break the set into train and val randomly
'''
import csv
import os
from Model import Dataset, classes
import random
from shutil import copyfile

random.seed(42)

dir_name = "Data/images"

assert (os.path.exists(dir_name) and os.path.isdir(dir_name)) and os.listdir(
    dir_name), "Images folder empty or not found"

assert os.path.exists(dir_name + "/labels.csv"), "labels.csv file not found"


class Dataset_Splitter(Dataset):
    '''
    Balanced validation is very important so you have no choice >:)
    '''

    def __init__(self, dir_name, val_percent=0.15):
        self.target_train_dir = 'Data/dataset/train/'
        self.target_val_dir = 'Data/dataset/val/'
        self.val_percent = val_percent
        self.path_data = Dataset(dir_name).path_data
        self.imgnumber_of_smallest_classes = len(self.path_data)  # lazy initialization
        self.smallest_class = None
        self.data_per_class = []
        self.val_data = []
        self.train_data = []
        self.read_data()
        self.split_data()

    def read_data(self):
        labels = [d[1] for d in self.path_data]
        self.number_of_classes = max(labels) + 1
        assert self.number_of_classes > 0, "0 classes found"
        for i in range(0, self.number_of_classes):
            class_data = [d for d in self.path_data if d[1] == i]
            self.data_per_class.append(class_data)
            if len(class_data) < self.imgnumber_of_smallest_classes:
                self.imgnumber_of_smallest_classes = len(class_data)

    def split_data(self):
        val_per_class = int(self.imgnumber_of_smallest_classes * self.val_percent)
        for i in range(0, self.number_of_classes):
            all_indx = [i for i in range(len(self.data_per_class[i]))]
            idx = random.sample(range(len(self.data_per_class[i])), val_per_class)
            not_idx = [a for a in all_indx if a not in idx]
            l = list(map(self.data_per_class[i].__getitem__, idx))
            not_l = list(map(self.data_per_class[i].__getitem__, not_idx))
            self.val_data.append(l)
            self.train_data.append(not_l)

    def write_dataset_folder(self):
        os.system('rm -r Data/dataset')
        os.makedirs(os.path.dirname(self.target_train_dir + "labels.csv"), exist_ok=True)
        os.makedirs(os.path.dirname(self.target_val_dir + "labels.csv"), exist_ok=True)
        for cls in self.train_data:
            for d in cls:
                data = []
                data.append(d)
                with open(self.target_train_dir + "labels.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerows(data)
                    copyfile(d[0], self.target_train_dir + os.path.basename(d[0]))
        for cls in self.val_data:
            for d in cls:
                data = []
                data.append(d)
                with open(self.target_val_dir + "labels.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerows(data)
                    copyfile(d[0], self.target_val_dir + os.path.basename(d[0]))

    def print_stats(self):
        print("There are {} of classes indentified".format(len(self.data_per_class)))
        for i in range(len(self.data_per_class)):
            print("Class {}  \t {}       \t     {} of val images \t {} of train images".format(i, classes[i],
                                                                                               len(self.val_data[i]),
                                                                                               len(self.train_data[i])))


if __name__ == '__main__':
    split_data = Dataset_Splitter(dir_name)
    split_data.write_dataset_folder()
    split_data.print_stats()
