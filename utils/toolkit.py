import os
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )
    OA_all, AA_mean_all, Kappa_all, AA_all = my_cal_oa_aa_ka(y_true,y_pred)
    all_acc["total_oa"]=OA_all
    all_acc["total_aa"]=AA_mean_all
    all_acc["total_kappa"]=Kappa_all
    all_acc["total_class"]=AA_all
    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
        OA_inc, AA_mean_inc, Kappa_inc, AA_inc = my_cal_oa_aa_ka(y_true[idxes],y_pred[idxes])
        all_acc[label+"_oa"]=OA_inc
        all_acc[label+"_aa"]=AA_mean_inc
        all_acc[label+"t_kappa"]=Kappa_inc
        all_acc[label+"_class"]=AA_inc
    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )
    if len(idxes) == 0:
        OA_old, AA_mean_old, Kappa_old, AA_old = 0,0,0,0
    else:
        OA_old, AA_mean_old, Kappa_old, AA_old = my_cal_oa_aa_ka(y_true[idxes],y_pred[idxes])
    all_acc["old_oa"]=OA_old
    all_acc["old_aa"]=AA_mean_old
    all_acc["old_kappa"]=Kappa_old
    all_acc["old_class"]=AA_old
    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )
    OA_new, AA_mean_new, Kappa_new, AA_new = my_cal_oa_aa_ka(y_true[idxes],y_pred[idxes])
    all_acc["new_oa"]=OA_new
    all_acc["new_aa"]=AA_mean_new
    all_acc["new_kappa"]=Kappa_new
    all_acc["new_class"]=AA_new
    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def list2dict(list):
    dict = {}
    for l in list:
        s = l.split(' ')
        id = int(s[0])
        cls = s[1]
        if id not in dict.keys():
            dict[id] = cls
        else:
            raise EOFError('The same ID can only appear once')
    return dict

def text_read(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            lines[i] = line.strip('\n')
    return lines

def my_cal_oa_aa_ka(true,pred):
        Kappa = cohen_kappa_score(np.array(true).reshape(-1, 1), np.array(pred).reshape(-1, 1))
        OA = (pred==true).mean()
        num_label = np.unique(true)
        N = len(num_label)
        avec = []
        for i in range(N):
            class_idx = np.where(true==num_label[i])
            class_pred = pred[class_idx]
            class_true = true[class_idx]
            class_acc = (class_true==class_pred).mean()
            avec.append(class_acc)
        mean_class = np.mean(np.asarray(avec))
        #print('aa: ',mean_class)
        return OA,mean_class,Kappa,avec
