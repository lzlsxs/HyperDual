import numpy as np
import torch
import os
import torch.utils.data as Data
from utils.preprocess import data_preprocss

class My_Dataset(Data.TensorDataset):
    def __init__(self,x_data,y_label):
        super(My_Dataset,self).__init__(x_data,y_label)
        self.transform = None
        self.labels = y_label
        self.images = x_data

class DataManager(object):
    def __init__(self, datasets, init_cls,increment,patch_size, pca_num,class_order):
        self._class_order = class_order
        self.all_datasets = datasets
        self.patch_size = patch_size
        self.pca_num = pca_num
        self._setup_data()
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)
        t_total = 0
        for i in range(len(self._increments)):
            t_total = t_total + self._increments[i]
        assert t_total == len(self._class_order), "incremental classes and dataset classes number not mismatch."
        assert len(self._temp_cls) == len(self._class_order), "class order is wrong."

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(self, indices, source,s_num=0):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

    

        data, targets = [], []
        for source_idx in indices:
            idx = self._class_order[source_idx]
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1, shot_num=s_num)

            data.append(class_data)
            class_targets = torch.full(class_targets.shape, source_idx)
            targets.append(class_targets)



        data, targets = torch.cat(data), torch.cat(targets)
        sel_dataset = My_Dataset(data,targets)
        return sel_dataset


    def _setup_data(self):
        #all_datasets =['Pavia','Indian','Salinas','Houston','Botswana']
        #dataname_to_path = {'Indian':'./data/IndianPine.mat', 'Houston':'./data/Houston.mat','Salinas':'./data/Salinas.mat','Pavia':'./data/Pavia.mat','Botswana':'./data/Botswana_my.mat','LongKou':'./data/WHU_Hi_LongKou_my.mat'}
        dataname_to_path = {'Indian':'/root/Desktop/我的网盘/datasets/IndianPine.mat', 'Houston':'/root/Desktop/我的网盘/datasets/Houston.mat','Salinas':'/root/Desktop/我的网盘/datasets/Salinas.mat','Pavia':'/root/Desktop/我的网盘/datasets/Pavia.mat','Botswana':'/root/Desktop/我的网盘/datasets/Botswana_my.mat','ZY1E':'/root/Desktop/我的网盘/datasets/zy1e.mat','ZH011':'/root/Desktop/我的网盘/datasets/zh011.mat','ZH012':'/root/Desktop/我的网盘/datasets/zh012.mat'}
        tsk_offset = 0
        all_train_data =[]
        all_train_label=[]
        all_test_data=[]
        all_test_label=[]
        for i in range(len(self.all_datasets)):
            data_path = os.path.expanduser(dataname_to_path[self.all_datasets[i]])
            x_train_band, y_train, x_test_band,y_test, num_classes,_ = data_preprocss(data_path, self.patch_size, self.pca_num)
            y_train = y_train + tsk_offset
            y_test = y_test + tsk_offset
            tsk_offset += num_classes
            x_data_train=torch.from_numpy(x_train_band).type(torch.FloatTensor) 
            y_label_train=torch.from_numpy(y_train).type(torch.LongTensor)
            x_data_test=torch.from_numpy(x_test_band).type(torch.FloatTensor) 
            y_label_test=torch.from_numpy(y_test).type(torch.LongTensor)
            all_train_data.append(x_data_train)
            all_train_label.append(y_label_train)
            all_test_data.append(x_data_test)
            all_test_label.append(y_label_test)
        all_train_data = torch.cat(all_train_data)
        all_train_label = torch.cat(all_train_label)
        all_test_data = torch.cat(all_test_data)
        all_test_label = torch.cat(all_test_label)
        self._train_data, self._train_targets = all_train_data, all_train_label
        self._test_data, self._test_targets = all_test_data, all_test_label
        self._temp_cls = [i for i in range(len(np.unique(all_train_label)))]

    def _select(self, x, y, low_range, high_range, shot_num=0):
        idxes_all = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        if shot_num != 0:
            idxes = np.random.choice(idxes_all,shot_num,replace=False)
        else:
            idxes = idxes_all
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))
