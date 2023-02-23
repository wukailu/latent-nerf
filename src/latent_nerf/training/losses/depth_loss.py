import torch
from torch import nn
from src.latent_nerf.configs.train_config import GuideConfig
from src.utils import load_dpt, infer_depth


class DepthLoss(nn.Module):
    # TODO: there might be a problem with RGB and BGR
    def __init__(self, cfg: GuideConfig):
        super().__init__()
        self.cfg = cfg
        self.model = load_dpt(self.cfg.dpt_model_path)
        self.depth_threshold = 0
        from torchmetrics import PearsonCorrCoef
        self.metric = PearsonCorrCoef()

    def forward(self, rgbs: torch.Tensor, depths: torch.Tensor):
        """
        @param rgbs: rendered rgbs, [N,3,H0,W0], in range [0, 1], larger - further, less than threshold => background
        @param depths: rendered depths, [N, H1, W1]
        """
        self.metric.to(rgbs.device)
        mask = depths > self.depth_threshold
        if not mask.any():
            return 0
        # print("rgbs shape ", rgbs.shape, depths.shape)
        down_rgbs = torch.nn.functional.interpolate(rgbs, (384, 384), mode='bilinear')
        depths_high_resolution = infer_depth(self.model, down_rgbs)
        # print("depths_high_resolution", depths_high_resolution.shape)
        ref_depths = torch.nn.functional.interpolate(depths_high_resolution[None], size=depths.shape[-2:], mode='bilinear')[0][mask]
        depths = depths[mask]
        return -self.metric(ref_depths.to(depths), depths)  # pearson 1-> related -1 -> not related
