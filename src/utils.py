import random
import os
from pathlib import Path

import numpy as np
import torch


def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis

    # res[(phis < front)] = 0
    # 这里其实写错了，但是由于初始化就是0，其他部分写对了这里错了不执行也没关系，最终结果是对的
    res[(phis >= (2 * np.pi - front / 2)) & (phis < front / 2)] = 0

    # res[(phis >= front) & (phis < np.pi)] = 1
    res[(phis >= front / 2) & (phis < (np.pi - front / 2))] = 1

    # res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    res[(phis >= (np.pi - front / 2)) & (phis < (np.pi + front / 2))] = 2

    # res[(phis >= (np.pi + front))] = 3
    res[(phis >= (np.pi + front / 2)) & (phis < (2 * np.pi - front / 2))] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def tensor2numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return tensor


def make_path(path: Path) -> Path:
    path.mkdir(exist_ok=True, parents=True)
    return path


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def find_best_gpus(num_gpu_needs=1):
    import subprocess as sp
    gpu_ids = []
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [(int(x.split()[0]), i) for i, x in enumerate(memory_free_info) if i not in gpu_ids]
    print('memories left ', memory_free_values)
    memory_free_values = sorted(memory_free_values)[::-1]
    gpu_ids = [k for m, k in memory_free_values[:num_gpu_needs]]
    return gpu_ids


def load_dpt(model_check_point="src/DPT/weights/dpt_hybrid-midas-501f0c75.pt"):
    from dpt.models import DPTDepthModel
    model = DPTDepthModel(
        path=model_check_point,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    return model.eval().cpu()


def infer_depth(model, img, shape_384=False):
    from torchvision.transforms import Compose, Resize, Normalize
    """
    @param img: Numpy array, (1, 151, 151, 3), [0, 1.0]
    @param model: dpt_hybird model from load_dpt
    """
    device = img.device
    if device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()
    model.to(device)

    net_w = net_h = 384
    transform = Compose(
        [
            Resize([net_h, net_w]),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    img_input = transform(img)

    # compute
    with torch.no_grad():
        sample = img_input.to(device)

        if device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        prediction = model.forward(sample)  # [N, H, W]
        if not shape_384:
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )[:, 0]  # [N, 1, H, W]
    return prediction  # [N, H, W]


# unit test passed
if __name__ == "__main__":
    import glob
    from PIL import Image
    import numpy as np

    img_names = glob.glob(os.path.join("src/DPT/input", "*"))
    model = load_dpt()
    for idx, name in enumerate(img_names):
        img = torch.tensor(np.array(Image.open(name))).permute((2, 0, 1))[None, :3] / 255.
        print(img.shape)
        pred = infer_depth(model, img.cuda())
        print(pred.shape)
        from DPT.util.io import write_depth
        write_depth(f"tmp{idx}", pred.cpu().squeeze().numpy(), bits=2, absolute_depth=False)
