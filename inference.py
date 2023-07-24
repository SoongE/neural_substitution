import hydra
from omegaconf import DictConfig
from timm.models import load_checkpoint

from src.initial_setting import init_seed, init_distributed, init_logger, cuda_setting
from src.models import *
from src.utils import ObjectFactory


@hydra.main(config_path="configs", config_name="sub_resnet", version_base="1.3")
def main(cfg: DictConfig) -> None:
    import torch
    cfg.gpus = [8]
    cuda_setting(cfg.gpus)
    init_distributed(cfg)
    init_seed(cfg.train.seed + cfg.local_rank)

    device = torch.device(f'cuda:{cfg.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    cfg.model.model_name = 'resnet18_Sub333'
    init_logger(cfg)
    cfg.dataset.augmentation.prefetcher = True
    cfg.dataset.augmentation.pin_mem = True
    cfg.train.batch_size = 5

    factory = ObjectFactory(cfg)
    model = factory.create_model()

    path = {
        'resnet18': '/home/seungmin/dmount/neural_substitution/runs/imageNet_resnet18/resnet18_20230424-154018/model_best.pth.tar',
        'resnet18_SubInception': '/home/seungmin/dmount/neural_substitution/runs/imageNet_resnet18_SubInception/resnet18_SubInception_20230424-153144/model_best.pth.tar',
        'resnet18_Sub333': '/home/seungmin/dmount/neural_substitution/runs/resnet18_Sub333/model_best.pth.tar'
    }
    load_checkpoint(model, path[cfg.model.model_name])
    input = torch.load('/home/seungmin/dmount/neural_substitution/runs/input.pth').to(device)

    model.eval()
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with torch.no_grad():
        out = model.forward_features(input)
        print('OUT: ', out.shape)
        if out.dim() == 5:
            out = out.sum(-1)

        deploy(model, BasicBlockSub)
        n_reparam = sum(p.numel() for p in model.parameters() if p.requires_grad)

        re_out = model.forward_features(input)
        print('RE-OUT: ', re_out.shape)

        print("Difference of feature map: ", end='')
        print(((out - re_out) ** 2).sum())

        avgpool = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        out = model.fc(avgpool(out))
        re_out = model.fc(avgpool(re_out))

        print("Difference of probability: ", end='')
        print(((out - re_out) ** 2).sum())

        pre = out.argmax(-1)
        re_pre = re_out.argmax(-1)

        print("Difference of prediction: ", end='')
        print(pre, re_pre)
        print((pre.eq(re_pre).float().mean()))

        print("Number of parameters: ", end='')
        print(f"{n_param} / {n_reparam}")


if __name__ == "__main__":
    main()
