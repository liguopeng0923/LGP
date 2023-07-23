import torch
import argparse
from mmcv.image import tensor2imgs
from sewar.full_ref import psnrb
from piq import LPIPS,TVLoss,InformationWeightedSSIMLoss,BRISQUELoss
from adv.utils.transforms import imdenormalize
import pyiqa
import mmcv
import os
class PerceptualDistance(object):
    def __init__(self,device):
        
        self.l_inf_avg = 0
        self.brisque_avg = 0
        self.psnrb_avg = 0
        self.tv_avg = 0
        self.lpips_avg = 0
        self.iwssim_avg = 0

        self.l_inf_sum = 0
        self.brisque_sum = 0
        self.psnrb_sum = 0
        self.tv_sum = 0
        self.lpips_sum = 0
        self.iwssim_sum = 0

        self.count = 0
        self.device = device
        
        self.lpips = LPIPS(replace_pooling=True)
        self.tvloss = TVLoss()
        self.brisque = BRISQUELoss()
        self.iwssim = InformationWeightedSSIMLoss()
        
    def cal_perceptual_distances(self, references, perturbed, img_metas):
        # l_p norm
        N = references.shape[0]
        img_norm_cfg = img_metas[0]['img_norm_cfg']
        
        org = torch.clamp(imdenormalize(references,**img_norm_cfg) / 255,min=0,max=1)
        adv = torch.clamp(imdenormalize(perturbed,**img_norm_cfg) / 255,min=0,max=1)
        adv = torch.where(torch.isnan(adv), torch.full_like(adv, 0), adv)
        noise = (adv - org).flatten(start_dim=1)
        
        l_inf_sum = torch.sum(torch.norm(noise, p=float('inf'), dim=-1)) 
        
        lpips_sum = self.lpips(org, adv)
        iwssim_sum =  self.iwssim(org, adv)
        brisque_sum = self.brisque(adv)
        tv_sum = self.tvloss(adv)
        
        org_numpy = tensor2imgs(references.detach(), **img_norm_cfg)
        adv_numpy = tensor2imgs(perturbed.detach(), **img_norm_cfg)
        
        psnrb_sum = sum([psnrb(img1, img2+1e-10) for img1,img2 in zip(org_numpy,adv_numpy)])
        
        return l_inf_sum/N, iwssim_sum/N,psnrb_sum/N,lpips_sum/N, brisque_sum/N,  tv_sum/N

    def update(self, l_inf,iwssim,psnrb, lpips, brisque, tv,n=1):
        
        self.l_inf_sum += l_inf * n
        self.brisque_sum += brisque * n 
        self.psnrb_sum += psnrb * n 
        self.tv_sum += tv * n
        self.lpips_sum += lpips * n
        self.iwssim_sum += iwssim * n
        
        self.count += n

        self.l_inf_avg = self.l_inf_sum / self.count
        self.brisque_avg = self.brisque_sum / self.count
        self.psnrb_avg = self.psnrb_sum / self.count
        self.tv_avg =  self.tv_sum / self.count
        self.lpips_avg = self.lpips_sum / self.count
        self.iwssim_avg = self.iwssim_sum / self.count
        
    def images_metric(self, references, perturbed, img_metas):
        return self.cal_perceptual_distances(references, perturbed, img_metas)
    
def fid(org_path,adv_path):
    fid_metric = pyiqa.create_metric('fid')
    score = fid_metric(org_path, adv_path)
    print(score)
    return score

if __name__ == '__main__':
    """Parse parameters."""
    parser = argparse.ArgumentParser(
        description='Get the score of FID')
    parser.add_argument('org_path', help='orginal images file path')
    parser.add_argument('adv_path', help='adversarial examples file path')
    parser.add_argument('work_dir', help='adversarial examples file path')
    args = parser.parse_args()
    mmcv.dump(fid(args.org_path,args.adv_path),os.path.join(args.work_dir, "fid.json"))
    
    # CUDA_VISIBLE_DEVICES=0 python /remote-home/liguopeng/paper/CVPR2023/mmrotate/adv_perceptible_metric.py /remote-home/liguopeng/paper/CVPR2023/mmrotate/data/split_ss_dota/val/images /remote-home/liguopeng/paper/CVPR2023/cvpr2023/TIP/RF/TOG/images /remote-home/liguopeng/paper/CVPR2023/cvpr2023/TIP/RF/TOG/