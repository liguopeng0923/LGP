import torch

from torch.autograd import Variable
from mmrotate.core.bbox import rbbox_overlaps

from ..builder import ATTACK_METHODS,build_optimizer,build_losses
from ..utils.DWT import *
from ..utils.FBS import *
from ..utils.transforms import *
from .attackBase import attackBase

@ATTACK_METHODS.register_module()
class LGP(attackBase):

    def __init__(self,
                 num_iteration: int =50,
                 model = None,
                 alpha: float = 1,
                 beta: float = 1,
                 gamma: float = 1,
                 miu: float = 0.1,
                 single_stage: bool = False,
                 adv_labels = -1,
                 fore_scale:float = 1.5,
                 wh_scale: float = 3,
                 iou_cond:int = 5,
                 score_cond = 5,
                 optimizer = None,
                 iouLoss = None,
                 clsLoss = None,
                 regLoss = None,
                 advPerLoss = None,
                 wave: str = 'haar'
                 ) -> None:
        
        super(LGP, self).__init__(num_iteration,
                 model,
                 alpha,
                 beta,
                 gamma,
                 miu,
                 single_stage,
                 adv_labels,
                 fore_scale,
                 wh_scale,
                 iou_cond,
                 score_cond,
                 optimizer,
                 iouLoss,
                 clsLoss,
                 regLoss,
                 advPerLoss,
                 wave)
        
    def forward(self,data) -> torch.Tensor:
        device = self.device
        model = self.model
        model.eval()
        mode = model.mode
        single_stage = self.single_stage
        for parm in model.parameters():
            parm.requires_grad = False
        images = data['img'].data
        batch = images.shape[0]
        img_metas = data['img_metas']
        gt_bboxes = data['gt_bboxes']
        # The preprocessing of data
        org = images.clone().to(device = device)
        proposal_bboxes, proposal_scores = model(img = images, img_metas = img_metas,rescale=False)
        
        num_classes = proposal_scores[0].shape[1]
        if mode == "hq":
            target_bboxes, target_scores= self._get_targets(
                gt_bboxes,proposal_bboxes, proposal_scores, self.iou_cond, self.score_cond
            )
            target_bboxes = [bboxes.reshape(-1,bboxes.shape[-1]) for bboxes in target_bboxes]
            target_scores = [scores.reshape(-1,scores.shape[-1]) for scores in target_scores]
        else:
            target_bboxes = proposal_bboxes
            target_scores = proposal_scores
        
        
        # The perceptibility limition of adversarial perturbation 
        fbs = FBS(img_metas,device,self.fore_scale)
        background_clean,background_weights = fbs(org,gt_bboxes,"background",weight = True)
        advPerLoss = build_losses(self.advPerLoss)
        regLoss = build_losses(self.regLoss)
        clsLoss = build_losses(self.clsLoss)
        iouLoss = build_losses(self.iouLoss)
        modifier = Variable(images, requires_grad=True).to(device)
        optimizer = build_optimizer([modifier], self.optimizer)
        zeros_imgs = imnormalize(torch.zeros_like(modifier).to(device),**img_metas[0]['img_norm_cfg'])
        max_imgs = imnormalize(255 * torch.ones_like(modifier).to(device),**img_metas[0]['img_norm_cfg'])
        for step in range(self.num_iteration):
            if single_stage:
                all_bboxes,all_scores = model(img = modifier, img_metas = img_metas,rescale=False)
                bboxes_unique, scores_unique = self._get_targets(
                target_bboxes,all_bboxes,all_scores, 1, 1)
                bboxes = [bbox.reshape(-1,5) for bbox in bboxes_unique]
                scores = [score.reshape(-1,num_classes) for score in scores_unique]
            else:
                if mode == "roi":
                    bboxes,scores = model(img = modifier, img_metas = img_metas,rescale=False)
                else:
                    bboxes,scores = model(img = modifier, img_metas = img_metas,proposals = target_bboxes,rescale=False)
            
            # The perceptibility limition of adversarial perturbation 
            adv_scores = torch.cat(scores,dim=0).to(device=device)
            adv_bboxes = torch.cat(bboxes,dim=0).to(device=device)
            
                
            if not (adv_scores.numel() and adv_bboxes.numel()):
                break
            
            if self.adv_labels == -1:
                adv_labels = adv_scores.new_full((adv_scores.shape[0],),
                                        num_classes-1,
                                        dtype=torch.long)
            else:
                adv_labels = adv_scores.new_full((adv_scores.shape[0],),
                                        self.adv_labels,
                                        dtype=torch.long)

            current_bboxes, current_scores = self._get_targets(
                gt_bboxes,bboxes,scores, self.iou_cond, self.score_cond
            )
            
            iou_cost = [-iouLoss(current_bboxes[i].reshape(-1,5),gt_bboxes[i].repeat_interleave(current_bboxes[i].shape[1],0).reshape(-1,5)) + regLoss(current_bboxes[i].reshape(-1,5)[:,:2],gt_bboxes[i].repeat_interleave(current_bboxes[i].shape[1],0).reshape(-1,5)[:,:2]) for i in range(batch)]
            iou_cost = -sum(iou_cost)
            
            current_labels = [torch.argmax(scores,dim=-1) for scores in current_scores]
            success_gt_bboxes = []
            failed_gt_bboxes = []
            for i in range(batch):
                if step <= 1:
                    failed_gt_bboxes.append(gt_bboxes[i])
                    success_gt_bboxes.append([])
                # 第一种gt有框，都是背景，第二种gt没框
                background_label =  num_classes - 1
                mask1 = (current_labels[i] == background_label)
                mask1 = mask1.all(dim=-1).reshape(gt_bboxes[i].shape[0])
                mask2,_ = torch.sort(rbbox_overlaps(gt_bboxes[i],bboxes[i],mode = "iou"),dim = -1,descending = True)
                mask2 = torch.sum(mask2,dim=-1)
                mask2 = mask2 < 0.1
                mask = mask1 + mask2
                failed_gt_bboxes.append(gt_bboxes[i][~mask])
                success_gt_bboxes.append(gt_bboxes[i][mask])
            foreground_clean_failed,foreground_weights_failed = fbs(org,failed_gt_bboxes,"foreground",weight = True)
            foreground_adv_failed = fbs(modifier,failed_gt_bboxes,"foreground")
            
                
            background_adv = fbs(modifier,gt_bboxes,"background")
            
            _, first_label = torch.max(adv_scores, -1)
            first_active_cond = first_label != adv_labels
            self._bboxes[step] += first_active_cond.sum().detach().cpu()
            
            # remove the background bboxes
            if not first_active_cond.any():
                if step <= self.num_iteration // 2:
                    modifier.data = org * 0.1 + modifier * 0.9
                    continue
                else:
                    break
            if mode != "roi":
                adv_scores = adv_scores[first_active_cond]
                adv_bboxes = adv_bboxes[first_active_cond]
                adv_labels = adv_labels[first_active_cond]
                first_label = first_label[first_active_cond]
            # reg attack
            adv_bboxes_scale = adv_bboxes.clone()
            adv_bboxes_scale[:,2:4] = adv_bboxes_scale[:,2:4] * self.wh_scale
            reg_cost = regLoss(adv_bboxes_scale[:,2:4],adv_bboxes[:,2:4]) * batch
            
            # cls attack
            cls_cost = clsLoss(adv_scores,adv_labels)
            
            adv_per_cost = advPerLoss(background_clean,background_adv,weight = background_weights) + advPerLoss(foreground_clean_failed,foreground_adv_failed,weight = foreground_weights_failed) + 0.1 * torch.sum((modifier-org) * (modifier-org) * foreground_weights_failed)
            
            total_cost = self.alpha * reg_cost + self.beta * iou_cost + self.gamma * cls_cost + self.miu * adv_per_cost
            
            before = modifier.clone()
            optimizer.zero_grad()
            total_cost.backward()
            optimizer.step()
            after = modifier.clone()
            
            with torch.no_grad():
                mask = (after > zeros_imgs) & (after < max_imgs)
                modifier.data = torch.where(mask,after,before)
            
            modifier.grad.zero_()
            model.zero_grad()
            
            self._losses[0][step] += total_cost.detach().cpu()
            self._losses[1][step] += self.alpha * reg_cost.detach().cpu()
            self._losses[2][step] += self.beta * iou_cost.detach().cpu()
            self._losses[3][step] += self.gamma * cls_cost.detach().cpu()
            self._losses[4][step] += self.miu * adv_per_cost.detach().cpu()
            
            modifier.grad.zero_()
            model.zero_grad()
        
        mask1 = (modifier > zeros_imgs)
        modifier = torch.where(mask1,modifier,zeros_imgs)
        mask2 = (modifier < max_imgs)
        modifier = torch.where(mask2,modifier,max_imgs)
        model.mode = "last"
        bboxes,_ = model(img = modifier, img_metas = img_metas,rescale=False)
        model.mode=mode
        ious = rbbox_overlaps(torch.cat(bboxes,dim=0),torch.cat(gt_bboxes,dim=0))
        iou50 = (ious > 0.5).sum()
        iou75 = (ious > 0.75).sum()
        return modifier,iou50,iou75
        # return modifier * (1 - background_weights) + org * background_weights,iou50,iou75
