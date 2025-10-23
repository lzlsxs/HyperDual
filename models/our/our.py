from models.our.ETFHead import ETFHead,MLPFFNNeck
from models.base import BaseLearner
from utils.inc_net import get_convnet
import os
from torch import nn
import logging
from torch.utils.data import DataLoader
import numpy as np
from torch import optim
import torch
from torch.nn import functional as F
import torchvision.transforms.functional as F1
from utils.toolkit import  tensor2numpy
from tqdm import tqdm
import copy
from models.our.supcon import SupConLoss
from models.our.utils import KD_loss
from utils.data_manager import My_Dataset
from sklearn.metrics.pairwise import cosine_similarity
from torch.distributions.multivariate_normal import MultivariateNormal
from copy  import deepcopy


class NC_model(nn.Module):
    def __init__(self, args):
        super(NC_model, self).__init__()
        self.backbone = get_convnet(args, pretrained=False)
        feat_num = self.backbone.out_dim
        self.neck = MLPFFNNeck(in_channels=feat_num,out_channels=feat_num)
        self.head = ETFHead(num_classes=args["total_classes"], in_channels = feat_num)
        self.val_classes = 0
    
    @property
    def feature_dim(self):
        return self.back_bone.out_dim
    def update_fc(self,num_cls):
        self.val_classes= num_cls
    
    '''
    def extract_feat(self,img, stage='neck'):
        assert stage in ['backbone', 'neck', 'logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')
        if self.backbone is not None:
            feat1 = self.backbone(img)['features']
        else:
            feat1 = img
        if stage == 'backbone':
            return feat1
        feat2 = self.neck(feat1)
        if stage == 'neck':
            return feat2
        logits = self.head(feat2)[:,:self.val_classes]
        return logits
    '''
    
    def forward_backbone(self, x):
        return self.backbone(x)['features']
    
    def forward_neck(self,x):
        return self.neck(x)

    def forward_head(self,x):
        return self.head(x)[:,:self.val_classes]
    
    def forward(self, x):
        feat1 = self.backbone(x)['features']
        feat2 = self.neck(feat1)
        x = self.head(feat2)[:,:self.val_classes]
        output1 ={"features":feat2, "logits":x}
        return output1
    
    def _extract_vectors(self, loader_):
        self.backbone.eval()
        vectors2, targets2 = [], []
        for _inputs, _targets in loader_:
            _inputs = _inputs.cuda()
            feat1 = self.backbone(_inputs)['features']
            feat1 = tensor2numpy(feat1)
            _targets = _targets.numpy()
            vectors2.append(feat1)
            targets2.append(_targets)
        #return torch.cat(vectors2), torch.cat(targets2)
        return np.concatenate(vectors2), np.concatenate(targets2)
    
    

    def _extract_neck_vectors(self, loader_):
        self.backbone.eval()
        vectors2, targets2 = [], []
        for _inputs, _targets in loader_:
            _inputs = _inputs.cuda()
            feat1 = self.backbone(_inputs)['features']
            feat1 = self.neck(feat1)
            feat1 = tensor2numpy(feat1)
            _targets = _targets.numpy()
            vectors2.append(feat1)
            targets2.append(_targets)
        #return torch.cat(vectors2), torch.cat(targets2)
        return np.concatenate(vectors2), np.concatenate(targets2)


class Our(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = NC_model(self.args)
        self._protos = []
        self._vars_static=[]  #add
        self._means_static =[] #add
    
    def after_task(self):
        self._old_network = copy.deepcopy(self._network)
        for param in self._old_network.parameters():
            param.requires_grad = False
        #self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        if not self.args['resume']:
            if not os.path.exists(self.args["model_dir"]):
                os.makedirs(self.args["model_dir"])
            self.save_checkpoint("{}{}_seed{}".format(self.args["model_dir"],self.args["dataset"][0],self.args["seed"]))
    
    def get_train_loader(self):
        return self.train_loader, self.test_loader
    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        shot_num = self.args["shot_num"]
        if self._cur_task == 0:
            shot_num = 0
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", s_num=shot_num
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", s_num=0
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if (not self.args['resume']) or self.args['NCM']:
            self._build_protos(train_dataset) 
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        
    

    def _train(self, train_loader, test_loader):
        resume = self.args['resume']  # set resume=True to use saved checkpoints
        if self._cur_task == 0:
            #resume = True
            if resume:
                self._network.load_state_dict(torch.load("{}{}_seed{}_{}.pkl".format(self.args["model_dir"],self.args["dataset"][0],self.args["seed"],self._cur_task))["model_state_dict"], strict=False)
                for param in self._network.backbone.parameters():
                    param.requires_grad = False
                self._network.backbone.eval()
            self._network.to(self._device)
            if hasattr(self._network, "module"):
                self._network_module_ptr = self._network.module
            if not resume:
                #self._train_backbone_new(train_loader, test_loader)
                self._init_train(train_loader, test_loader)
        else:
            resume = self.args['resume']
            if resume:
                self._network.load_state_dict(torch.load("{}{}_seed{}_{}.pkl".format(self.args["model_dir"],self.args["dataset"][0],self.args["seed"],self._cur_task))["model_state_dict"], strict=False)
            self._network.to(self._device)
            if hasattr(self._network, "module"):
                self._network_module_ptr = self._network.module
            if self._old_network is not None:
                self._old_network.to(self._device)
            if not resume:
                self._update_representation(train_loader, test_loader)
         
    

    def _build_protos(self, train_dataset):
        for class_idx in range(self._known_classes, self._total_classes):
            cls_idxes = (train_dataset.labels == class_idx)
            cls_imgs = train_dataset.images[cls_idxes]
            cls_labels = train_dataset.labels[cls_idxes]
            idx_dataset = My_Dataset(cls_imgs,cls_labels)
            idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False)
            vectors = self._network._extract_vectors(idx_loader)[0]
            class_mean = np.mean(vectors, axis=0) # vectors.mean(0)
            self._protos.append(torch.tensor(class_mean).to(self._device))
            self._means_static.append(torch.tensor(class_mean).to(self._device))
            #feat_var = vectors.var(0)
            #self._vars_static.append(torch.tensor(feat_var).to(self._device))
            feat_cov = torch.tensor(np.cov(vectors.T))
            feat_cov = feat_cov.float() + torch.eye(feat_cov.shape[0])*(1e-6)
            #feat_cov = feat_cov.float()
            self._vars_static.append(feat_cov.to(self._device))

            # neck_vectors = self._network._extract_neck_vectors(idx_loader)[0]
            # neck_mean = np.mean(neck_vectors, axis=0)
            # neck_proto = torch.tensor(neck_mean).to(self._device)
            # neck_proto = neck_proto.unsqueeze(0)
            # self._network.head.test_etf_vec[:,class_idx] = self._network.head.pre_logits(neck_proto)
        # temp_proto=torch.stack(self._protos)
        # for class_idx in range(self._known_classes, self._total_classes):
        #     self._network.head.etf_vec[:,class_idx] = self._network.head.pre_logits(temp_proto)[class_idx]
    
    def _train_backbone_new(self,train_loader, test_loader):
        optimizer = optim.SGD(self._network.backbone.parameters(), momentum=0.9, lr=self.args["lrate"], weight_decay=self.args["weight_decay"])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lr_decay"])
        prog_bar = tqdm(range(self.args["epochs"]))
        back_out_num = self._network.backbone.out_dim
        total_classes = self._total_classes
        temp_head = nn.Linear(back_out_num,total_classes*2).to(self._device)
        optimizer = optim.SGD([{'params':self._network.backbone.parameters()},{'params':temp_head.parameters()}], momentum=0.9, lr=self.args["lrate"], weight_decay=self.args["weight_decay"])
        ce_loss = nn.CrossEntropyLoss()
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            temp_head.train()
            losses = 0.0
            correct = 0
            total = 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                n = inputs.shape[1]
                arr = torch.arange(n).mul(-1).add(n-1)
                inputs_reverse = inputs[:,arr,:,:]
                targets_reverse = targets*2 + 1
                targets_old = targets * 2
                inputs_new = torch.cat([inputs, inputs_reverse], dim=0)
                targets_new = torch.cat([targets_old,targets_reverse])
                feat = self._network.forward_backbone(inputs_new)
                preds = temp_head(feat)
                loss = ce_loss(preds,targets_new)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds1 = torch.max(preds, dim=1)
                correct += preds1.eq(targets_new.expand_as(preds1)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            print(train_acc)
        for param in self._network.backbone.parameters():
            param.requires_grad = False
        self._network.backbone.eval()

    def _train_backbone(self,train_loader, test_loader):
        optimizer = optim.SGD(self._network.backbone.parameters(), momentum=0.9, lr=self.args["lrate"]*0.01, weight_decay=self.args["weight_decay"])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lr_decay"])
        prog_bar = tqdm(range(self.args["epochs"]))
        scl = SupConLoss()
        etf_feat = self._network.head.etf_vec.t()
        etf_label = torch.arange(self.args["total_classes"]).cuda()
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            losses = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                feat1 = self._network.forward_backbone(inputs)
                feat1 = torch.cat((feat1,etf_feat),dim=0)
                feat1 = feat1.unsqueeze(1)
                targets = torch.cat((targets,etf_label))
                loss = scl(feat1,targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                print(loss)

            scheduler.step()
        for param in self._network.backbone.parameters():
            param.requires_grad = False
        self._network.backbone.eval()

    def _init_train(self, train_loader, test_loader):
        scl = SupConLoss()
        optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.args["lrate"], weight_decay=self.args["weight_decay"])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lr_decay"])
        prog_bar = tqdm(range(self.args["epochs"]))
        etf_feats = self._network.head.etf_vec.t()
        etf_labels = torch.arange(self.args["total_classes"]).cuda()
        lamda = 1
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (inputs, targets) in enumerate(train_loader):
                loss = 0
                loss1 = 0
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                back_feat = self._network.forward_backbone(inputs)
                if self.args["sp_scl"]:
                    # input_hflip = F1.hflip(inputs)
                    # input_vflip = F1.vflip(inputs)
                    # back_feat_hflip = self._network.forward_backbone(input_hflip)
                    # back_feat_vflip = self._network.forward_backbone(input_vflip)
                    # pse_inputs = deepcopy(inputs)
                    # pse_targets = deepcopy(targets) + self.args["total_classes"]
                    # n = pse_inputs.shape[1]
                    # arr = torch.arange(n).mul(-1).add(n-1)
                    # pse_inputs = pse_inputs[:,arr,:,:]
                    # # for pse_idx in range(pse_inputs.shape[0]):
                    # #     pse_targets[pse_idx] = pse_idx + self.args["total_classes"]
                    # back_feat_pse = self._network.forward_backbone(pse_inputs)
                    # back_feat_mix = torch.cat([back_feat,back_feat_hflip,back_feat_vflip,back_feat_pse],dim=0)
                    # target_mix = torch.cat([targets,targets,targets,pse_targets])
                    # # scl_feats = back_feat_mix.unsqueeze(1)
                    # # loss1 = 0.1*scl(scl_feats,target_mix)
                    # loss1 = self._network.head.dr_constrative(back_feat_mix, target_mix)
                    # lamda = 0.9

                    # n = inputs.shape[1]
                    # arr = torch.arange(n).mul(-1).add(n-1)
                    # pse_inputs = deepcopy(inputs)[:,arr,:,:]
                    # back_feat_pse = self._network.forward_backbone(pse_inputs)
                    # pse_targets = deepcopy(targets) + self.args["total_classes"]
                    # input_hflip = F1.hflip(inputs)
                    # input_vflip = F1.vflip(inputs)
                    # back_feat_hflip = self._network.forward_backbone(input_hflip)
                    # back_feat_vflip = self._network.forward_backbone(input_vflip)
                    # back_feat = torch.cat([back_feat,back_feat_hflip,back_feat_vflip],dim=0)
                    # #scl_feats = back_feat.unsqueeze(1)
                    # targets = torch.cat([targets,targets,targets])
                    # #loss1 = 0.1*scl(scl_feats,targets)
                    # back_feat_mix = torch.cat([back_feat, back_feat_pse],dim=0)
                    # targets_mix = torch.cat([targets,pse_targets])
                    # loss1 = self._network.head.dr_constrative(back_feat_mix, targets_mix)
                    # #loss += loss1
                    # #lamda = epoch / self.args["epochs"]
                    # lamda = 0.9


                    input_hflip = F1.hflip(inputs)
                    input_vflip = F1.vflip(inputs)
                    back_feat_hflip = self._network.forward_backbone(input_hflip)
                    back_feat_vflip = self._network.forward_backbone(input_vflip)
                    back_feat = torch.cat([back_feat,back_feat_hflip,back_feat_vflip],dim=0)
                    temp_targets = torch.arange(targets.shape[0]).cuda()
                    targets = torch.cat([targets,targets,targets])

                    # n = inputs.shape[1]
                    # arr = torch.arange(n).mul(-1).add(n-1)
                    # pse_inputs = deepcopy(inputs)[:,arr,:,:]
                    # back_feat_pse = self._network.forward_backbone(pse_inputs)
                    # pse_targets = temp_targets + targets.shape[0]
                    # back_feat_mix = torch.cat([back_feat,back_feat_hflip,back_feat_vflip,back_feat_pse],dim=0)
                    targets_mix = torch.cat([temp_targets,temp_targets,temp_targets])
                    #loss1 = self._network.head.dr_constrative(back_feat, targets_mix)
                    scl_feats = back_feat.unsqueeze(1)
                    loss1 = scl(scl_feats,targets_mix) 
                    lamda = 0.99

                    # inputs_90 = torch.rot90(inputs,1,[2,3])
                    # inputs_180 = torch.rot90(inputs,2,[2,3])
                    # inputs_270 = torch.rot90(inputs,3,[2,3])
                    # back_feat_90 = self._network.forward_backbone(inputs_90)
                    # back_feat_180 = self._network.forward_backbone(inputs_180)
                    # back_feat_270 = self._network.forward_backbone(inputs_270)
                    # back_feat = torch.cat([back_feat,back_feat_90,back_feat_180,back_feat_270],dim=0)
                    # targets = torch.cat([targets,targets,targets,targets])
                    # scl_feats = back_feat.unsqueeze(1)
                    # loss1 = 0.1*scl(scl_feats,targets)
                    # loss += loss1
                if self.args["sp_scl"]:
                    loss += self.args["stage1"] * self.args["stage2"] *loss1

                feats = self._network.forward_neck(back_feat)
                if self.args["sp_scl"]:
                    loss += (1-self.args["stage1"]) * self.args["stage2"] * self._network.head.dr_constrative(feats,targets)
                logits = self._network.forward_head(feats)
                if self.args["add_kd"]:
                    loss3 = self._network.head.dr_loss_l1(feats,targets)
                else:
                    loss3 = self._network.head.dr_loss_l2(feats,targets)
                #loss = (1-lamda)*loss1 + lamda * loss2
                #indian 0.001 1 1
                #houston 0.018 0.018 1
                #pavia 0.01 0.01 1
                # if self.args["sp_scl"]:
                #     if self.args["dataset"][0] == "Indian":
                #         loss = 0.001* loss1 + 1*loss2 + loss3
                #     else:
                #         loss = 0.01* loss1 + 0.01*loss2 + loss3
                # else:
                #     loss = loss3
                loss += loss3

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 25 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)
        #self._network.eval()
        for param in self._network.backbone.parameters():
            param.requires_grad = False
        self._network.backbone.eval()
    

    def _update_representation(self, train_loader, test_loader):
        scl = SupConLoss()
        optimizer = optim.SGD(self._network.neck.parameters(), lr=self.args["lrate"], momentum=0.9, weight_decay=self.args["weight_decay"])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lr_decay"])
        if self.args["ssm_proto"]:
            proto_feats = torch.stack(self._protos)
            proto_labels = torch.arange(self._known_classes).cuda()
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.neck.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (inputs, targets) in enumerate(train_loader):
                if self.args["ssm_gaussian"]:
                    # proto_feats1 = torch.normal(proto_means, proto_vars)
                    # proto_feats = torch.cat((proto_feats0, proto_feats1),dim=0)
                    gaussian_labels = torch.arange(self._known_classes).cuda()
                    samples_list=[]
                    for proto_idx in range(self._known_classes):
                        mean_idx = self._means_static[proto_idx]
                        #cov_idx = self._vars_static[proto_idx].float() + torch.eye(mean_idx.shape[0]).to(self._device)*(1e-6)
                        cov_idx = self._vars_static[proto_idx]
                        m = MultivariateNormal(loc=mean_idx, covariance_matrix=cov_idx)
                        sam_idx = m.sample()
                        samples_list.append(sam_idx)
                    gaussian_feats = torch.stack(samples_list)
                
                if self.args["ssm_proto"] and self.args["ssm_gaussian"]:
                    replay_feats = torch.cat((proto_feats,gaussian_feats),dim=0)
                    replay_labels = torch.cat([proto_labels,gaussian_labels])
                elif self.args["ssm_proto"]:
                    replay_feats = proto_feats
                    replay_labels = proto_labels
                elif self.args["ssm_gaussian"]:
                    replay_feats = gaussian_feats
                    replay_labels = gaussian_labels
                else:
                    replay_feats = None
                    replay_labels = None
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                backbone_feats = self._network.forward_backbone(inputs)
                weight = None
                # weight_1 = F.cosine_similarity(backbone_feats[None,:,:], proto_feats[:,None,:],dim=-1).max(0)[0]
                # weight_2 = F.cosine_similarity(proto_feats[None,:,:], backbone_feats[:,None,:],dim=-1).max(0)[0]
                # weight = torch.cat((weight_1, weight_2))
                if replay_feats is not None:
                    feats = torch.cat((backbone_feats,replay_feats), dim=0)
                    labels = torch.cat((targets,replay_labels))
                else:
                    feats = backbone_feats
                    labels = targets
                neck_feats = self._network.neck(feats)

                if self.args["add_kd"]:
                    loss = self._network.head.dr_loss_l1(neck_feats,labels,weight)
                else:
                    loss = self._network.head.dr_loss_l2(neck_feats,labels,weight)
                
                if self.args["add_kd"]:
                    old_neck_feats = self._old_network.neck(feats)
                    loss_kd = KD_loss(neck_feats,old_neck_feats,self.args["T"])
                    # contr_neck_feats = torch.cat((neck_feats,old_neck_feats),dim=0)
                    # contr_targets = torch.cat([labels,labels])
                    # loss_kd = self._network.head.dr_constrative(contr_neck_feats,contr_targets)
                    #loss_kd = torch.mean(torch.sqrt(torch.sum( torch.pow(old_neck_feats-neck_feats,2),dim=-1 )))
                    loss = loss + self.args["kd_lamda"] * loss_kd
                logits = self._network.head(neck_feats[:backbone_feats.shape[0]])[:,:self._total_classes]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 25 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)
