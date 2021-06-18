import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib.pyplot import imread
from tqdm import tqdm

import flow_transforms
import models
from post_process import post_process

parser = argparse.ArgumentParser(description='Test Optical Flow',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', dest='dataset', default='KITTI', choices=['KITTI_occ', 'humanflow'],
                    help='test dataset')
parser.add_argument('--div-flow', default=20, type=float,
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
parser.add_argument('--no-norm', action='store_true',
                    help='don\'t normalize the image')
parser.add_argument('--pretrained', metavar='PTH',
                    default=None, help='path to pre-trained model')
parser.add_argument('--save-name', dest='save_name', type=str, default=None,
                    help='flow network architecture. Options: Name for saving results')
parser.add_argument('--output-dir', dest='output_dir',
                    metavar='DIR', default=None, help='path to output flo')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TAG_CHAR = 'PIEH'.encode()


def main():
    global args
    args = parser.parse_args()

    if args.output_dir is None:
        raise argparse.ArgumentError

    if args.pretrained is not None:
        network_data = torch.load(args.pretrained)
        args.arch = network_data['arch']
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](data=network_data).cuda()
        if 'div_flow' in network_data.keys():
            args.div_flow = network_data['div_flow']
    else:
        model = models.pwc_dc_net('models/pwc_net.pth.tar').cuda()

    model.eval()

    if args.no_norm:
        input_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255])
        ])
    else:
        input_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
        ])

    for img_paths, _, seg_path in tqdm(make_dataset(args.data)):
        img1 = input_transform(255*imread(img_paths[0])[:, :, :3]).squeeze()
        img2 = input_transform(255*imread(img_paths[1])[:, :, :3]).squeeze()

        # Resize Image from 640*640 to 448x448
        size = 448
        img1 = F.upsample(img1.unsqueeze(0), (size, size),
                          mode='bilinear').squeeze()
        img2 = F.upsample(img2.unsqueeze(0), (size, size),
                          mode='bilinear').squeeze()

        input_var = torch.cat([img1, img2]).unsqueeze(0).to(device)

        # compute output, output size is input/4
        output = model(input_var)

        # resize the image to 160*160 and smooth the background
        output = torch.Tensor(
            post_process(
                output.squeeze().cpu().detach().numpy(),
                seg_path
            )
        ).unsqueeze(0).to(device)

        output_path = img_paths[0].replace(args.data, args.output_dir)
        output_path = output_path.replace('/composition/', '/')
        os.system('mkdir -p '+output_path[:-10])

        output_path = output_path.replace('.png', '.flo')
        output_path = output_path.replace('/flow/', '/')
        print(f"output path: {output_path}", output.shape)

        flow_write(output_path,  output.cpu()[0].data.numpy()[
                   0],  output.cpu()[0].data.numpy()[1])

    print("Finished generating and exporting output.")


def load_flo(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert(202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (w, h, 2))
    return data2D


def make_dataset(dir, phase='test'):
    '''Will search for triplets that go by the pattern '[name]_img1.ppm  [name]_img2.ppm    [name]_flow.flo' '''
    images = []
    #print(f"Making dataset from: {os.path.join(dir, phase+'/*/flow/*.flo')}")
    for img1 in sorted(glob.glob(os.path.join(dir, phase+'/*/composition/*.png'))):
        img2 = img1[:-9] + \
            str(int(img1.split('/')[-1][:-4])+1).zfill(5) + '.png'

        if int(img1.split('/')[-1][:-4]) % 10 == 9:
            continue

        if not (os.path.isfile(os.path.join(dir, img1)) and os.path.isfile(os.path.join(dir, img2))):
            continue

        seg_mask = img1.replace('/flow/', '/segm_EXR/').replace('.flo', '.exr')

        images.append([[img1, img2], None, seg_mask])

    return images


def flow_write(filename, uv, v=None):
    """ Write optical flow to file.
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        uv_ = np.array(uv)
        assert(uv_.ndim == 3)
        if uv_.shape[0] == 2:
            u = uv_[0, :, :]
            v = uv_[1, :, :]
        elif uv_.shape[2] == 2:
            u = uv_[:, :, 0]
            v = uv_[:, :, 1]
        else:
            raise TypeError('Wrong format for flow input')
    else:
        u = uv

    assert(u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:, np.arange(width)*2] = u
    tmp[:, np.arange(width)*2 + 1] = v
    tmp.astype(np.float16).tofile(f)
    f.close()


if __name__ == '__main__':
    main()
