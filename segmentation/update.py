import time
import torch
import copy
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from eval_utils import evaluate

import myseg.bisenet_utils
from myseg.bisenet_utils import OhemCELoss,BackCELoss,CriterionPixelPair,CriterionPixelRegionPair,ContrastLoss,ContrastLossLocal,CriterionPixelPairG,CriterionPixelPairSeq
from myseg.magic import MultiEpochsDataLoader
import numpy as np
#from segmentation_models_pytorch.losses import JaccardLoss,DiceLoss,FocalLoss,LovaszLoss,SoftBCEWithLogitsLoss


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
        self.trainloader, self.testloader,self.trainloader_eval = self.train_val_test(dataset, list(idxs))
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

        trainloader_eval = MultiEpochsDataLoader(DatasetSplit(dataset, idxs_train),
                                            batch_size=1, num_workers=self.args.num_workers,
                                            shuffle=False, drop_last=False, pin_memory=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=1, num_workers=self.args.num_workers,
                                shuffle=False)

        return trainloader, testloader,trainloader_eval

    @torch.no_grad()
    def get_protos(self,model,global_round):
        args = self.args
        model.eval()
        tmp_ = []
        label_list =  []
        label_mask_list = []
        for batch_idx, (images, labels) in enumerate(self.trainloader_eval):
            images, labels = images.to(self.device), labels.to(self.device)

            if args.model == 'bisenetv2':
                logits, feat_head, *logits_aux = model(images)

            _,_,h,w = feat_head.size()
            labels_2 = F.interpolate(logits.float(),size=(h,w),mode='bilinear')
            labels_2 = torch.softmax(labels_2,dim=1)
            props, labels_2 = torch.max(labels_2,dim=1)
#                        print(props.max())
#                        print(props.min())
            mask_ = props<0.8
            labels_2[mask_]=255

            feat_head = feat_head.unsqueeze(1)

            labels = labels.unsqueeze(1)
            labels = F.interpolate(labels.float(),size=(h,w),mode='nearest')
            labels = labels.unsqueeze(1)

#            print(labels.size())
#            print(labels_2.size())
#            exit()


            labels_2 = labels_2.unsqueeze(1).unsqueeze(1)

            #labels_2[labels!=255]=labels


            labels = torch.where(labels.float()!=255,labels.float(),labels_2.float())
            unique_l = torch.unique(labels.cpu()).numpy().tolist()
            label_list.extend(unique_l)
            one_hot_ = torch.zeros(args.num_classes).to(self.device)
            for ll in unique_l:
                ll = int(ll)
                if ll !=255:
                    one_hot_[ll]=1
            label_mask_list.append(one_hot_)
            

            class_ = torch.arange(args.num_classes).to(self.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            weight_ = class_ == labels
            weight_ = weight_/(weight_.sum(3,keepdim=True).sum(4,keepdim=True)+1e-5)
            out = weight_*feat_head
            out = out.sum(-1).sum(-1)
            tmp_.append(out)
        tmp_ = torch.cat(tmp_,0)
        tmp_ =  tmp_.permute(1,0,2)
            
#        print(tmp_.size())
#        tmp_ = sum(tmp_)/len(tmp_)
        label_mask_ = torch.stack(label_mask_list,1)
        return tmp_,label_list,label_mask_


    def update_weights(self, model, global_round,prototypes=None,proto_mask=None):

        # Set mode to train model
        model.train()
        epoch_loss = []



        # Set optimizer and lr_scheduler for the local updates
        args = self.args


        if args.distill or args.fedprox_mu >0:
            global_model = copy.deepcopy(model)
            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            criteria_distill_pi = CriterionPixelPairSeq(args,temperature=args.temp_dist)
            criteria_distill_pa =CriterionPixelRegionPair(args)
            pixel_seq = []
            


        if args.is_proto:
            criteria_contrast = ContrastLoss(args)
            global_model = copy.deepcopy(model)
            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False


#            criteria_proaux = [OhemCELoss(0.7) for _ in range(4)]  # num_aux_heads=4
 

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
            elif args.losstype == 'lovasz':
                criteria_pre = LovaszLoss('multiclass',ignore_index=255)
                criteria_aux = [LovaszLoss('multiclass',ignore_index=255) for _ in range(4)]  # num_aux_heads=4
 
            elif args.losstype == 'dice':
                criteria_pre = DiceLoss('multiclass',args.num_classes,ignore_index=255)
                criteria_aux = [DiceLoss('multiclass',args.num_classes,ignore_index=255) for _ in range(4)]  # num_aux_heads=4
            elif args.losstype == 'focal':

                criteria_pre = FocalLoss('multiclass',alpha=0.25,ignore_index=255)
                criteria_aux = [FocalLoss('multiclass',alpha=0.25,ignore_index=255) for _ in range(4)]  # num_aux_heads=4
             
            elif args.losstype == 'bce':
                criteria_pre = SoftBCEWithLogitsLoss(ignore_index=255)
                criteria_aux = [SoftBCEWithLogitsLoss(ignore_index=255) for _ in range(4)]  # num_aux_heads=4
             
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

                    
                    labels_ = labels

                    if args.losstype == 'bce':
                        cl_ = torch.arange(args.num_classes)
                        cl_ = cl_.unsqueeze(0).unsqueeze(2).unsqueeze(2)
                        cl_ = cl_.to(labels_.device)
                        labels_ = labels_.unsqueeze(1) ==cl_
                        labels_ = labels_.float()

#                    print(logits.size())
#                    print(labels.size())
#                    exit()
                    loss_pre = criteria_pre(logits, labels_)
                    loss_aux = [crit(lgt, labels_) for crit, lgt in zip(criteria_aux, logits_aux)]
                    loss = loss_pre + sum(loss_aux)
                else:
                    exit('Error: unrecognized model')

                ##########
                if args.is_proto and global_round>= args.proto_start_epoch:

                    _,_,h,w = feat_head.size()

                    labels_1 = labels_.unsqueeze(1)
                    labels_1 = F.interpolate(labels_1.float(),size=(h,w),mode='nearest')
                    labels_1 = labels_1.squeeze(1)
                    #print(feat_head.size())
                    #print(labels_1.size())
                    #print(prototypes.size())
                    #print(proto_mask.size())
                    #exit()
                    if args.kmean_num>0:

                        proto_mask_tmp = proto_mask.sum(1)<1
                    else:
                        proto_mask_tmp = proto_mask<1
                    for ii, bo in enumerate(proto_mask_tmp):
                        if bo:
                            labels_1[labels_1==ii]=255

                    loss_con = criteria_contrast(feat_head,labels_1,prototypes,proto_mask)
                    loss_con_item = loss_con.item()
                    loss_ce = loss.item()
                    loss +=args.con_lamb*loss_con 
                    
                    if args.pseudo_label and global_round>=args.pseudo_label_start_epoch:
                        device = prototypes.device
                        with torch.no_grad():
                            logits_t, feat_head_t, *logits_aux_t = global_model(images)
                        labels_2 = F.interpolate(logits_t.float(),size=(h,w),mode='bilinear')
                        labels_2 = torch.softmax(labels_2,dim=1) 
                        props, labels_2 = torch.max(labels_2,dim=1)
#                        print(props.max())
#                        print(props.min())


                        mask_ = props<0.8
                        labels_2[mask_]=255
             

                        for ii, bo in enumerate(proto_mask_tmp):
                            if bo:
                                labels_2[labels_2==ii]=255
                            
                        loss_con_2 = criteria_contrast(feat_head,labels_2,prototypes,proto_mask)
                        loss_con_2_item = loss_con_2.item()
                        loss +=args.con_lamb*loss_con_2
                        
####################


####################

                else:
                    loss_ce = loss.item()
                    loss_con_item=0

                ########
                if args.fedprox_mu >0:
                    proximal_term = 0.0
                    for w, w_t in zip(model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    loss += (args.fedprox_mu / 2) * proximal_term

                if args.distill:
                    loss_1_item = loss.item()
                    with torch.no_grad():
                        logits_t, feat_head_t, *logits_aux_t = global_model(images)                 
                    if args.distill_lamb_pi>0 and args.is_proto and global_round>= args.proto_start_epoch:
                        loss_pi,pixel_seq=criteria_distill_pi(feat_head,feat_head_t.detach(),pixel_seq)
                        loss_pi = args.distill_lamb_pi *loss_pi
                               
                        loss+=loss_pi
                        loss_pi_item = loss_pi.item()
                    else:
                        loss_pi_item=0
                    if args.distill_lamb_pa>0 and args.is_proto and global_round>= args.proto_start_epoch:
                        loss_pa=args.distill_lamb_pa*criteria_distill_pa(feat_head,feat_head_t.detach(),prototypes,proto_mask)
                        loss+=loss_pa
                        loss_pa_item = loss_pa.item()
                    else:
                        loss_pa_item=0


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  # update params
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
        if args.distill:
            print('Loss_CE:{:.6f} | loss_pi:{:.6f} | loss_pa:{:.6f}'.format(loss_1_item,loss_pi_item,loss_pa_item))
        
        if args.is_proto:
            if global_round>= args.proto_start_epoch:

                if args.pseudo_label:
                    print('Loss_CE:{:.6f} | loss_contrast:{:.6f} loss_pseudo: {:.6f}'.format(loss_ce,loss_con_item,loss_con_2_item))
                else:
                    print('Loss_CE:{:.6f} | loss_contrast:{:.6f}'.format(loss_ce,loss_con_item))
            else:
                print('Loss_CE:{:.6f}'.format(loss_ce))

            

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
