import pyrallis
import os
from src.utils import find_best_gpus

gpu_id = find_best_gpus(1)[0]
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

from src.latent_nerf.configs.train_config import TrainConfig
from src.latent_nerf.training.trainer import Trainer



@pyrallis.wrap()
def main(cfg: TrainConfig):
    trainer = Trainer(cfg)
    if cfg.log.eval_only:
        trainer.full_eval()
    else:
        trainer.train()

if __name__ == '__main__':
    main()