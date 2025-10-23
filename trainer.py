import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np
import random


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    patch_size_list = copy.deepcopy(args["patch_size"])
    #stage_list = copy.deepcopy(args["stage1"])
    init_list = copy.deepcopy(args["init_cls"])
    #args["patch_size"] = patch_size_list[0]
    #pca_list = copy.deepcopy(args["pca_num"])
    for patch_size in patch_size_list:
        args["patch_size"] = patch_size
    #for stage1 in stage_list:
    #    args["stage1"] = stage1
    #for pca_t in pca_list:
    #    args["pca_num"] = pca_t
        for seed in seed_list:
            args["seed"] = seed
            args["device"] = device
            print(args["patch_size"])
            #print(args["pca_num"])
            print(args["seed"])
            print('+'*50)
            _train(args)


def _train(args):

    _set_random(args["seed"])
    #_set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["init_cls"],
        args["increment"],
        args["patch_size"],
        args["pca_num"],
        args["class_order"]
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve,nme_curve = {"top1": []},{"top1": []}
    for task in range(data_manager.nb_tasks):
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()
        if args["NCM"]:
            print("nme_grouped: {}".format(nme_accy["grouped"]))
            nme_curve["top1"].append(nme_accy["top1"])
            print("nme top1 curve: {}".format(nme_curve["top1"]))
        else:
            print("cnn_grouped: {}".format(cnn_accy["grouped"]))
            cnn_curve["top1"].append(cnn_accy["top1"])
            print("cnn top1 curve: {}".format(cnn_curve["top1"]))

def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
