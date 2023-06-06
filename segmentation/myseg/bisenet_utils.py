import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from myseg.bisenetv2 import BiSeNetV2


def set_model_bisenetv2(args,num_classes):
    net = BiSeNetV2(args,num_classes) # num_classes = 19

    # if not args.finetune_from is None:
    #     logger.info(f'load pretrained weights from {args.finetune_from}')
    #     net.load_state_dict(torch.load(args.finetune_from, map_location='cpu'))

    # if cfg.use_sync_bn: net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

    # net.cuda()
    # net.train()

    # criteria_pre = OhemCELoss(0.7)
    # criteria_aux = [OhemCELoss(0.7) for _ in range(4)]  # num_aux_heads=4
    # return net, criteria_pre, criteria_aux

    return net


def set_optimizer(model, args):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        #  wd_val = cfg.weight_decay
        wd_val = 0
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': wd_val},
            {'params': lr_mul_wd_params, 'lr': args.lr * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': args.lr * 10},
        ]
        # params_list = [
        #     {'params': wd_params, },
        #     {'params': nowd_params, 'weight_decay': wd_val},
        #     {'params': lr_mul_wd_params, 'lr': current_lr * 10},
        #     {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': current_lr * 10},
        # ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    # optim = torch.optim.SGD(
    #     params_list,
    #     lr=current_lr,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay,
    # )
    return optim

class BackCELoss(nn.Module):
    def __init__(self, args, ignore_lb=255):
        super(BackCELoss, self).__init__()
        self.ignore_lb = ignore_lb
        self.class_num = args.num_classes
        self.criteria = nn.NLLLoss(ignore_index=ignore_lb, reduction='mean')
    def forward(self, logits, labels):
        total_labels = torch.unique(labels)
        new_labels = labels.clone()
        probs = torch.softmax(logits,1)
        fore_ = []
        back_ = []
        
        for l in range(self.class_num):
            if l in total_labels:
                fore_.append(probs[:,l,:,:].unsqueeze(1))
            else: 
                back_.append(probs[:,l,:,:].unsqueeze(1))
        Flag=False
        if not  len(fore_)==self.class_num:
            fore_.append(sum(back_))
            Flag=True
        
        for i,l in enumerate(total_labels):
            if Flag :
                new_labels[labels==l]=i
            else: 
                if l!=255:
                    new_labels[labels==l]=i
            
        probs  =torch.cat(fore_,1)
        logprobs = torch.log(probs+1e-7)
        return self.criteria(logprobs,new_labels.long())




class OhemCELoss(nn.Module):
    '''
    Feddrive: We apply OHEM (Online Hard-Negative Mining) [56], selecting 25%
    of the pixels having the highest loss for the optimization.
    '''

    def __init__(self, thresh, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        # n_min = labels[labels != self.ignore_lb].numel() // 16
        n_min = int(labels[labels != self.ignore_lb].numel() * 0.25)
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


#################################################

class CriterionPixelPair(nn.Module):
    def __init__(self, args,temperature=0.1,ignore_index=255, ):
        super(CriterionPixelPair, self).__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.args= args

    def pair_wise_sim_map(self, fea_0, fea_1):
        C, H, W = fea_0.size()

        fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
        fea_1 = fea_1.reshape(C, -1).transpose(0, 1)
        
        sim_map_0_1 = torch.mm(fea_0, fea_1.transpose(0, 1))
        return sim_map_0_1

    def forward(self, feat_S, feat_T):
        #feat_T = self.concat_all_gather(feat_T)
        #feat_S = torch.cat(GatherLayer.apply(feat_S), dim=0)
        B, C, H, W = feat_S.size()

        device = feat_S.device
        patch_w = 2
        patch_h = 2
        #maxpool = nn.MaxPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        maxpool = nn.AvgPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        feat_S = maxpool(feat_S)
        feat_T= maxpool(feat_T)
        feat_S = F.normalize(feat_S, p=2, dim=1)
        feat_T = F.normalize(feat_T, p=2, dim=1)
        
        sim_dis = torch.tensor(0.).to(device)
        for i in range(B):
            s_sim_map = self.pair_wise_sim_map(feat_S[i], feat_S[i])
            t_sim_map = self.pair_wise_sim_map(feat_T[i], feat_T[i])

            p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
            p_t = F.softmax(t_sim_map / self.temperature, dim=1)

            sim_dis_ = F.kl_div(p_s, p_t, reduction='batchmean')
            sim_dis += sim_dis_
        sim_dis = sim_dis / B 
        return sim_dis
######################################################

class CriterionPixelPairSeq(nn.Module):
    def __init__(self, args,temperature=0.1,ignore_index=255, ):
        super(CriterionPixelPairSeq, self).__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.args= args

    def pair_wise_sim_map(self, fea_0, fea_1):
        C, H, W = fea_0.size()

        fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
        fea_1 = fea_1.reshape(C, -1).transpose(0, 1)
        
        sim_map_0_1 = torch.mm(fea_0, fea_1.transpose(0, 1))
        return sim_map_0_1

    def forward(self, feat_S, feat_T, pixel_seq):
        #feat_T = self.concat_all_gather(feat_T)
        #feat_S = torch.cat(GatherLayer.apply(feat_S), dim=0)
        B, C, H, W = feat_S.size()

        device = feat_S.device
        patch_w = 2
        patch_h = 2
        #maxpool = nn.MaxPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        maxpool = nn.AvgPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        feat_S = maxpool(feat_S)
        feat_T= maxpool(feat_T)
        feat_S = F.normalize(feat_S, p=2, dim=1)
        feat_T = F.normalize(feat_T, p=2, dim=1)

        feat_S = feat_S.permute(0,2,3,1).reshape(-1,C)
        feat_T = feat_T.permute(0,2,3,1).reshape(-1,C)

#        split_T = torch.split(feat_T,1,dim=0)
        split_T = feat_T
        idx = np.random.choice(len(split_T),4000,replace=False)
        
        split_T = split_T[idx]
        split_T = torch.split(split_T,1,dim=0)
        pixel_seq.extend(split_T)
        if len(pixel_seq)>20000:
            del pixel_seq[:len(pixel_seq)-20000]
        

        proto_mem_ = torch.cat(pixel_seq,0)
        s_sim_map = torch.matmul(feat_S,proto_mem_.T)
        t_sim_map = torch.matmul(feat_T,proto_mem_.T)


        p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
        p_t = F.softmax(t_sim_map / self.temperature, dim=1)

        sim_dis = F.kl_div(p_s, p_t, reduction='batchmean')
        return sim_dis,pixel_seq
######################################################

class CriterionPixelPairG(nn.Module):
    def __init__(self, args,temperature=0.1,ignore_index=255, ):
        super(CriterionPixelPairG, self).__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.args= args

    def pair_wise_sim_map(self, fea_0, fea_1):
        C, H, W = fea_0.size()

        fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
        fea_1 = fea_1.reshape(C, -1).transpose(0, 1)
        
        sim_map_0_1 = torch.mm(fea_0, fea_1.transpose(0, 1))
        return sim_map_0_1

    def forward(self, feat_S, feat_T,proto_mem,proto_mask):
        #feat_T = self.concat_all_gather(feat_T)
        #feat_S = torch.cat(GatherLayer.apply(feat_S), dim=0)
        B, C, H, W = feat_S.size()

        device = feat_S.device
        patch_w = 2
        patch_h = 2
        #maxpool = nn.MaxPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        maxpool = nn.AvgPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        feat_S = maxpool(feat_S)
        feat_T= maxpool(feat_T)
        feat_S = F.normalize(feat_S, p=2, dim=1)
        feat_T = F.normalize(feat_T, p=2, dim=1)

        feat_S = feat_S.permute(0,2,3,1).reshape(-1,C)
        feat_T = feat_T.permute(0,2,3,1).reshape(-1,C)

        if self.args.kmean_num>0:
            C_,km_,c_ = proto_mem.size()
            proto_labels = torch.arange(C_).unsqueeze(1).repeat(1,km_)
            proto_mem_ = proto_mem.reshape(-1,c_)
            proto_mask = proto_mask.view(-1)
            proto_idx = torch.arange(len(proto_mask))
            proto_idx = proto_idx.to(device)
            sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
            proto_mem_ = proto_mem_[sel_idx]


        else:
            C_,c_ = proto_mem.size()
            proto_labels = torch.arange(C_)
            proto_mem_ = proto_mem
            proto_mask = proto_mask
            proto_idx = torch.arange(len(proto_mask))
            proto_idx = proto_idx.to(device)
            sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
            proto_mem_ = proto_mem_[sel_idx]

        s_sim_map = torch.matmul(feat_S,proto_mem_.T)
        t_sim_map = torch.matmul(feat_T,proto_mem_.T)


        p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
        p_t = F.softmax(t_sim_map / self.temperature, dim=1)

        sim_dis = F.kl_div(p_s, p_t, reduction='batchmean')
        return sim_dis
######################################################

class CriterionPixelRegionPair(nn.Module):
    def __init__(self,args, temperature=0.1,ignore_index=255, ):
        super(CriterionPixelRegionPair, self).__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.args = args

    def pair_wise_sim_map(self, fea_0, fea_1):
        C, H, W = fea_0.size()

        fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
        fea_1 = fea_1.transpose(0, 1)
        
        sim_map_0_1 = torch.mm(fea_0, fea_1)
        return sim_map_0_1

    def forward(self, feat_S, feat_T,proto_mem,proto_mask):
        #feat_T = self.concat_all_gather(feat_T)
        #feat_S = torch.cat(GatherLayer.apply(feat_S), dim=0)
        B, C, H, W = feat_S.size()

        device = feat_S.device
        
        if self.args.kmean_num>0:
            C_,U_,km_,c_ = proto_mem.size()
            proto_mem_ = proto_mem.reshape(-1,c_)
            proto_mask = proto_mask.unsqueeze(-1).repeat(1,1,km_).view(-1)
            proto_idx = torch.arange(len(proto_mask))
            proto_idx = proto_idx.to(device)
            sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
            proto_mem_ = proto_mem_[sel_idx]

        else:
            C_,U_,c_ = proto_mem.size()
            proto_mem_ = proto_mem.reshape(-1,c_)
            proto_mask = proto_mask.view(-1)
            proto_idx = torch.arange(len(proto_mask))
            proto_idx = proto_idx.to(device)
            sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
            proto_mem_ = proto_mem_[sel_idx]


        sim_dis = torch.tensor(0.).to(device)
        for i in range(B):
            s_sim_map = self.pair_wise_sim_map(feat_S[i], proto_mem_)
            t_sim_map = self.pair_wise_sim_map(feat_T[i], proto_mem_)

            p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
            p_t = F.softmax(t_sim_map / self.temperature, dim=1)

            sim_dis_ = F.kl_div(p_s, p_t, reduction='batchmean')
            sim_dis += sim_dis_
        sim_dis = sim_dis / B 
        return sim_dis

######################################################


def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
    def __init__(self, scale):
        '''inter pair-wise loss from inter feature maps'''
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.scale = scale

    def forward(self, preds_S, preds_T):
        feat_S = preds_S
        feat_T = preds_T
        feat_T.detach()

        total_w, total_h = feat_T.shape[2], feat_T.shape[3]
        patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
        maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
        loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
        return loss


class ContrastLoss(nn.Module):
    def __init__(self, args, ignore_lb=255):
        super(ContrastLoss, self).__init__()
        self.ignore_lb = ignore_lb
        self.args = args
        self.max_anchor = args.max_anchor
        self.temperature = args.temperature

    def _anchor_sampling(self,embs,labels):
        device = embs.device
        b_,c_,h_,w_ = embs.size()
        class_u = torch.unique(labels)
        class_u_num = len(class_u)
        if 255 in class_u:
            class_u_num =class_u_num - 1

        if class_u_num==0:
            return None,None

        num_p_c = self.max_anchor//class_u_num


        embs = embs.permute(0,2,3,1).reshape(-1,c_)

        labels = labels.view(-1)
        index_ = torch.arange(len(labels))
        index_ = index_.to(device)

        sampled_list = []
        sampled_label_list = []
        for cls_ in class_u:
       #     print(cls_)
            if cls_ != 255:
                mask_ = labels==cls_
                selected_index_ = torch.masked_select(index_,mask_)
                if len(selected_index_)>num_p_c:
                    sel_i_i = torch.arange(len(selected_index_))
                    sel_i_i_i = torch.randperm(len(sel_i_i))[:num_p_c]
                    sel_i = sel_i_i[sel_i_i_i]     
                    selected_index_ = selected_index_[sel_i]
       #             print(selected_index_.size())
                embs_tmp = embs[selected_index_]
                sampled_list.append(embs_tmp)
                sampled_label_list.append(torch.ones(len(selected_index_)).to(device)*cls_)
       # print('&'*10)
        sampled_list = torch.cat(sampled_list,0)
        sampled_label_list = torch.cat(sampled_label_list,0)

        return sampled_list,sampled_label_list


    def forward(self,embs,labels,proto_mem,proto_mask):
        device = proto_mem.device
        anchors,anchor_labels = self._anchor_sampling(embs,labels)
        if anchors is None:
            loss =torch.tensor(0).to(device)
            return loss 

        #print(anchors.size())
        #print(anchor_labels.size())
        #exit()

        if self.args.kmean_num>0:
            C_,km_,c_ = proto_mem.size()
            proto_labels = torch.arange(C_).unsqueeze(1).repeat(1,km_)
            proto_mem_ = proto_mem.reshape(-1,c_)
            proto_labels = proto_labels.view(-1)
            proto_mask = proto_mask.view(-1)
            proto_idx = torch.arange(len(proto_mask))
            proto_idx = proto_idx.to(device)
            sel_idx = torch.masked_select(proto_idx,proto_mask.bool())

            
            proto_labels =proto_labels.to(device)
            proto_mem_ = proto_mem_[sel_idx]
            proto_labels = proto_labels[sel_idx]
            proto_labels =proto_labels.to(device)

            
        else:
            C_,c_ = proto_mem.size()
            proto_labels = torch.arange(C_)
            proto_mem_ = proto_mem
            proto_labels = proto_labels
            proto_labels = proto_labels[sel_idx]
            proto_labels =proto_labels.to(device)
            proto_mask = proto_mask
            proto_idx = torch.arange(len(proto_mask))
            proto_idx = proto_idx.to(device)
            sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
            proto_mem_ = proto_mem_[sel_idx]
            proto_labels = proto_labels[sel_idx]
            proto_labels =proto_labels.to(device)

#        print(proto_mem_.size())
#        print(proto_labels.size())
#        exit()
        anchor_dot_contrast = torch.div(torch.matmul(anchors,proto_mem_.T),self.temperature)
        mask = anchor_labels.unsqueeze(1)==proto_labels.unsqueeze(0)
        mask = mask.float()
        mask = mask.to(device)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

        logits = anchor_dot_contrast - logits_max.detach()

        # mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        # logits_mask = torch.ones_like(mask).scatter_(1,
        #                                              torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
        #                                              0)

        # mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits) * mask
#        print(exp_logits.size())
#        print(logits.size())
#        print(neg_logits.size())
#        exit()
        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.mean()
        if torch.isnan(loss):
            print('!'*10)
            print(torch.unique(logits))
            print(torch.unique(exp_logits))
            print(torch.unique(neg_logits))
            print(torch.unique(log_prob))
            print(torch.unique(mask.sum(1)))
            print(mask)
            print(torch.unique(anchor_labels))
            print(proto_labels)
            print(torch.unique(proto_labels))
              
            exit()
#        print(loss)
#        print('*'*10)
        return loss





class ContrastLossLocal(nn.Module):
    def __init__(self, args, ignore_lb=255):
        super(ContrastLossLocal, self).__init__()
        self.ignore_lb = ignore_lb
        self.args = args
        self.max_anchor = args.max_anchor
        self.temperature = args.temperature

    def _anchor_sampling(self,embs,labels):
        device = embs.device
        b_,c_,h_,w_ = embs.size()
        class_u = torch.unique(labels)
        class_u_num = len(class_u)
        if 255 in class_u:
            class_u_num =class_u_num - 1

        if class_u_num==0:
            return None,None

        num_p_c = self.max_anchor//class_u_num


        embs = embs.permute(0,2,3,1).reshape(-1,c_)

        labels = labels.view(-1)
        index_ = torch.arange(len(labels))
        index_ = index_.to(device)

        sampled_list = []
        sampled_label_list = []
        for cls_ in class_u:
       #     print(cls_)
            if cls_ != 255:
                mask_ = labels==cls_
                selected_index_ = torch.masked_select(index_,mask_)
                if len(selected_index_)>num_p_c:
                    sel_i_i = torch.arange(len(selected_index_))
                    sel_i_i_i = torch.randperm(len(sel_i_i))[:num_p_c]
                    sel_i = sel_i_i[sel_i_i_i]     
                    selected_index_ = selected_index_[sel_i]
       #             print(selected_index_.size())
                embs_tmp = embs[selected_index_]
                sampled_list.append(embs_tmp)
                sampled_label_list.append(torch.ones(len(selected_index_)).to(device)*cls_)
       # print('&'*10)
        sampled_list = torch.cat(sampled_list,0)
        sampled_label_list = torch.cat(sampled_label_list,0)

        return sampled_list,sampled_label_list


    def forward(self,embs,labels,proto_mem,proto_mask,local_mem):
        device = proto_mem.device
        anchors,anchor_labels = self._anchor_sampling(embs,labels)
        if anchors is None:
            loss =torch.tensor(0).to(device)
            return loss 

        #print(anchors.size())
        #print(anchor_labels.size())
        #exit()

        if self.args.kmean_num>0:
            C_,U_,km_,c_ = proto_mem.size()
            proto_labels = torch.arange(C_).unsqueeze(1).unsqueeze(1).repeat(1,U_,km_)
            proto_mem_ = proto_mem.reshape(-1,c_)
            proto_labels = proto_labels.view(-1)
            proto_mask = proto_mask.unsqueeze(-1).repeat(1,1,km_).view(-1)
            proto_idx = torch.arange(len(proto_mask))
            proto_idx = proto_idx.to(device)
            sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
            proto_mem_ = proto_mem_[sel_idx]
            proto_labels = proto_labels[sel_idx]
            proto_labels =proto_labels.to(device)

        else:
            C_,U_,c_ = proto_mem.size()
            proto_labels = torch.arange(C_).unsqueeze(1).repeat(1,U_)
            proto_mem_ = proto_mem.reshape(-1,c_)
            proto_labels = proto_labels.view(-1)
            proto_mask = proto_mask.view(-1)
            proto_idx = torch.arange(len(proto_mask))
            proto_idx = proto_idx.to(device)
            sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
            proto_mem_ = proto_mem_[sel_idx]
            proto_labels = proto_labels[sel_idx]
            proto_labels =proto_labels.to(device)


        anchor_dot_contrast = torch.div(torch.matmul(anchors,proto_mem_.T),self.temperature)
        mask = anchor_labels.unsqueeze(1)==proto_labels.unsqueeze(0)
        mask = mask.float()
        mask = mask.to(device)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * mask

       ################## 
        C_,N_,c_= local_mem.size()
        local_labels = torch.arange(C_).unsqueeze(1).repeat(1,N_)
        local_mem = local_mem.reshape(-1,c_)
        local_labels = local_labels.view(-1)
        local_labels = local_labels.to(device)

        anchor_dot_contrast_l = torch.div(torch.matmul(anchors,local_mem.T),self.temperature)
        mask_l = anchor_labels.unsqueeze(1)==local_labels.unsqueeze(0)
        mask_l = mask_l.float().to(device)
        logits_l = anchor_dot_contrast_l - logits_max.detach()

        neg_logits = torch.exp(logits_l) * mask_l
        neg_logits = neg_logits.sum(1, keepdim=True)

######################################
        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.mean()
        if torch.isnan(loss):
            print('!'*10)
            print(torch.unique(logits))
            print(torch.unique(exp_logits))
            print(torch.unique(neg_logits))
            print(torch.unique(log_prob))

            exit()
#        print(loss)
#        print('*'*10)
        return loss





