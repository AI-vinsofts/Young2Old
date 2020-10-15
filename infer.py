import os
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from gan_module import Generator
import cv2 

parser = ArgumentParser()
parser.add_argument('--image_dir', default='image', help='The image directory')


value_depend=0.5
@torch.no_grad()
def main():
    args = parser.parse_args()
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir) if
                   x.endswith('.png') or x.endswith('.jpg')]
    import time
    a = time.time()
    model = Generator(ngf=32, n_residual_blocks=9)
    ckpt = torch.load('pretrained_model/state_dict.pth', map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(value_depend, value_depend, value_depend), std=(value_depend, value_depend, value_depend))
    ])
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    
    random.shuffle(image_paths)
    
    for i in range(3):
        a = time.time()
        img = Image.open(image_paths[i])
        img = trans(img).unsqueeze(0)
        print(time.time()-a)
        aged_face = model(img)
        aged_face = (aged_face.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
        ax[0, i].imshow((img.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0)
        ax[1, i].imshow(aged_face)
    plt.show()


if __name__ == '__main__':
    
    main()
