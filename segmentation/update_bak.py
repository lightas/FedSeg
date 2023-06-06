import time
import torch
import copy
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from eval_utils import evaluate

import myseg.bisenet_utils
from myseg.bisenet_utils import OhemCELoss,BackCELoss
from myseg.magic import MultiEpochsDataLoader


class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label)
        # pytorch warning and suggest below 
        return image.clone().detach().float(), label.clone().detach()


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader, self.testloader = self.train_val_test(dataset, list(idxs))
        #self.device = 'cuda' if torch.cuda.is_available() and not args.cpu_only else 'cpu'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, and test (80%, 20%)
        # idxs_train = idxs[:int(0.8*len(idxs))]
        # idxs_test = idxs[int(0.8*len(idxs)):]

        # split indexes for train, and test (100%, 50%)
        idxs_train = idxs[:]
        idxs_test = idxs[:int(0.5*len(idxs))]

        # try to change num_workers, to see if can speed up training. (num_workers=4 is better for training speed)

        # trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
        #                          batch_size=self.args.local_bs, num_workers=self.args.num_workers,
        #                          shuffle=True, drop_last=True, pin_memory=True)

        # use MultiEpochsDataLoader to speed up training
        trainloader = MultiEpochsDataLoader(DatasetSplit(dataset, idxs_train),
                                            batch_size=self.args.local_bs, num_workers=self.args.num_workers,
                                            shuffle=True, drop_last=True, pin_memory=True)

        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=2, num_workers=self.args.num_workers,
                                shuffle=False)

        return trainloader, testloader

    @torch.no_grad()
    def get_protos(self,model,global_round):
        args = self.args
        model.eval()
        tmp_ = []
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            if args.model == 'bisenetv2':
                logits, feat_head, *logits_aux = model(images)
                
            _,_,h,w = feat_head.size()
            feat_head = feat_head.unsqueeze(1)

            labels = labels.unsqueeze(1)
            labels = F.interpolate(labels.float(),size=(h,w),mode='nearest')
            labels = labels.unsqueeze(1)
            class_ = torch.arange(args.num_classes).to(labels.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            weight_ = class_ == labels
            weight_ = weight_/(weight_.sum(0,keepdim=True).sum(3,keepdim=True).sum(4,keepdim=True)+1e-5)
            out = weight_*feat_head
            out = out.sum(0).sum(-1).sum(-1)
            tmp_.append(out)
        tmp_ = sum(tmp_)/len(tmp_)
        return tmp_


    def update_weights(self, model, global_round,prototypes=None):
  

        if prototypes is not None:
            prototypes = prototypes.permute(1,0)

        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer and lr_scheduler for the local updates
        args = self.args

        if args.is_proto and not args.label_online_gen:
            model_record = copy.deepcopy(model)
            model_record.eval()

        if args.model == 'bisenetv2':
            optimizer = myseg.bisenet_utils.set_optimizer(model, args)
            if args.losstype=='ohem':
                criteria_pre = OhemCELoss(0.7)
                criteria_aux = [OhemCELoss(0.7) for _ in range(4)]  # num_aux_heads=4

            elif args.losstype=='ce':
                criteria_pre = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
                criteria_aux = [nn.CrossEntropyLoss(ignore_index=255, reduction='mean') for _ in range(4)]  # num_aux_heads=4
            elif args.losstype =='back':
                criteria_pre = BackCELoss(args)
                criteria_aux = [BackCELoss(args) for _ in range(4)]  # num_aux_heads=4
            else:
                raise ValueError('loss type is not defined')

        else:
            exit('Error: unrecognized model')

        # scheduler_dict = {
        #     'step': torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5),
        #     'lambda':torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / (len(self.trainloader)*max(1,args.local_ep))) ** 0.9)
        # }

        scheduler_dict = {
            'step': torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lambda x: (1 if global_round < 1000 else 0.1)),
            # 'step': torch.optim.lr_scheduler.LambdaLR(optimizer,
            #                                           lambda x: (1 if global_round < 1000 else (0.1 if global_round < 1200 else 0.01))),

            'poly': torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lambda x: (1 - x / (len(self.trainloader) * max(1, args.local_ep))) ** 0.9) #根据iter更新lr
        }
        lr_scheduler = scheduler_dict[args.lr_scheduler]

        # training
        start_time = time.time()
        for iter in range(args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                #print(labels.shape) # torch.Size([8, 512, 1024])

                # 计算loss
                if args.model == 'bisenetv2':

                    
                    
                    logits, feat_head, *logits_aux = model(images)

                    

                    if args.is_proto:
                        if not args.label_online_gen:
                            with torch.no_grad():
                                _, feat_head, *logits_aux_tmp = model_record(images)    
                                
                        _,h,w = labels.size()
                        feat_head = feat_head.permute(0,2,3,1)
                        feat_head = F.normalize(feat_head,-1)
                        new_label = torch.matmul(feat_head,prototypes)
                        new_label = new_label.permute(0,3,1,2).detach()
                        new_label = F.interpolate(new_label,size=(h,w),mode='nearest')
                        new_label = torch.argmax(new_label,dim=1)
                        labels_ = new_label.long()
                        
                    else:
                        labels_ = labels


#                    print(logits.size())
#                    print(labels.size())
#                    exit()
                    loss_pre = criteria_pre(logits, labels_)
                    loss_aux = [crit(lgt, labels_) for crit, lgt in zip(criteria_aux, logits_aux)]
                    loss = loss_pre + sum(loss_aux)
                else:
                    exit('Error: unrecognized model')

                loss.backward()
                optimizer.step()  # update params
                optimizer.zero_grad()
                batch_loss.append(loss.item())

                # 打印学习率
                print("Local Epoch: {}, batch_idx: {}, lr: {:.3e}".format(iter, batch_idx, lr_scheduler.get_lr()[0]))
                lr_scheduler.step() # lr_scheduler:poly,根据iter(每个batch)更新lr, (不是根据local_epoch更新)

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

            # 打印学习率
            #print("Local Epoch: {}, lr: {:.3e}".format(iter, lr_scheduler.get_lr()[0]))
            #print("Local Epoch: {}, lr: {:.3e}".format(iter, optimizer.param_groups[0]['lr'])) #两个打印学习率的方式都可以
            #lr_scheduler.step()

            if args.verbose:
                string = '| Global Round : {} | Local Epoch : {} | {} images\tLoss: {:.6f}'.format(
                    global_round, iter+1, len(self.trainloader.dataset), loss.item())
                print(string)

        # after training, print logs
        # strings = [
        #     '| Global Round : {} | Local Epochs : {} | {} images\tLoss: {:.6f}'.format(
        #     global_round, args.local_ep, len(self.trainloader.dataset), loss.item()),
        #     '\nLocal Train Run Time: {0:0.2f}s'.format(time.time()-start_time),
        #     ]

        # 不输出Local Train Run Time了
        strings = [
            '| Global Round : {} | Local Epochs : {} | {} images\tLoss: {:.6f}'.format(
                global_round, args.local_ep, len(self.trainloader.dataset), loss.item())
        ]
        print(''.join(strings))
        
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        confmat = evaluate(model, self.testloader, self.device, self.args.num_classes)
        # print(str(confmat)) # local test也输出信息
        return confmat.acc_global, confmat.iou_mean, str(confmat)


def test_inference(args, model, testloader):
    """ Returns the test accuracy and loss.
    """
    #device = 'cuda' if torch.cuda.is_available() and not args.cpu_only else 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    confmat = evaluate(model, testloader, device, args.num_classes)
    return confmat.acc_global, confmat.iou_mean, str(confmat)
