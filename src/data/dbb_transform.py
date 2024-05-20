import torch
import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

cifar_normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                       std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])


class PCALighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img
        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


def strong_train_preprocess(img_size):
    trans = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
        transforms.ToTensor(),
        PCALighting(0.1, imagenet_pca['eigval'], imagenet_pca['eigvec']),
        normalize,
    ])
    return trans


def standard_train_preprocess(img_size):
    trans = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    return trans


def val_preprocess(img_size):
    resize = 1 / 0.875 * img_size
    trans = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    return trans
