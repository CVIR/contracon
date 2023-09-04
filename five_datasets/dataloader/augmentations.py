from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
import torchvision.transforms as T
import torchvision.transforms.functional as Fi
class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.
        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128), random=False):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor, random),
            # GrayScalePolicy(),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor, random),            
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor, random),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor, random),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor, random),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor, random),
            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor, random),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor, random),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor, random),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor, random),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor, random),
            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor, random),
            

            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor, random),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor, random),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor, random),
            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor, random),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor, random),

            
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor, random),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor, random),
            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor, random),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor, random),

            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor, random),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor, random),
            
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor, random),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor, random)
        ]
        self.rand_policies = [
            T.ColorJitter(brightness=.5, hue=.3),
            T.RandomPerspective(distortion_scale=0.6, p=1.0),
            T.RandomRotation(degrees=(0, 180)),
            T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
            T.RandomInvert(),
            T.RandomPosterize(bits=2),
            T.RandomSolarize(threshold=192.0),
            T.RandomAdjustSharpness(sharpness_factor=2),
            T.RandomAutocontrast(),
            T.RandomEqualize()
        ]
        self.random = random
        if not  self.random:
            self.policy_idx = -1
    def __call__(self, img):
        # print("CALLING")
        if self.random:
            # print('OKAy')
            policy_idx = random.randint(0, len(self.policies) - 1)
            return self.policies[policy_idx](img)
        else:
            self.policy_idx += 1
            policy_idx = self.policy_idx
            # print("policy idx = ", self.policy_idx)
            return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SVHNPolicy(object):
    """ Randomly choose one of the best 25 Sub-policies on SVHN.
        Example:
        >>> policy = SVHNPolicy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     SVHNPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128), random=False):
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor, random),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor, random),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor, random),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor, random),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor, random),

            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor, random),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor, random),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor, random),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor, random),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor, random),

            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor, random),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor, random),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor, random),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor, random),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor, random),

            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor, random),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor, random),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor, random),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor, random),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor, random),

            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor, random),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor, random),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor, random),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor, random),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor, random)
        ]

        self.random = random
        if not  self.random:
            self.policy_idx = -1
    def __call__(self, img):
        if self.random:
            # print('OKAy')
            policy_idx = random.randint(0, len(self.policies) - 1)
            return self.policies[policy_idx](img)
        else:
            self.policy_idx += 1
            policy_idx = self.policy_idx
            # print("policy idx = ", self.policy_idx)
            return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"

class GrayScalePolicy(object):
    def __init__(self):
        self.func = T.Grayscale()
    
    def __call__(self, img):
        img = self.func(img)
        img = img.expand(3, -1, -1)
        return img
class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128), israndom=False):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }
        self.israndom = israndom
        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
                
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }


        func_notrand = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0], 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1]),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        if self.israndom:
            self.operation1 = func[operation1]
        else:
            self.operation1 = func_notrand[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        if self.israndom:
            self.operation2 = func[operation2]
        else:
            self.operation2 = func_notrand[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]


    def __call__(self, img):
        if self.israndom:
            if random.random() < self.p1:
                img = self.operation1(img, self.magnitude1)
            if random.random() < self.p2:
                img = self.operation2(img, self.magnitude2)
        else:
            img = self.operation2(img, self.magnitude2)
        return img