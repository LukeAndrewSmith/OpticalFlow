import argparse
import os
import glob
from tqdm import tqdm
import numpy as np
from imageio import imread, imsave
import skimage.transform

parser = argparse.ArgumentParser(description='Test Optical Flow',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--phase', dest='phase', type=str, default=None, help='path to dataset')
parser.add_argument('--pred-dir', dest='pred_dir', type=str, default=None,
                    help='path to prediction folder')
parser.add_argument('--output-dir', dest='output_dir', type=str, default=None,
                    help='path to save visualizations')


def main():
    global args
    args = parser.parse_args()
    test_list = make_real_dataset(args.data, phase=args.phase)
    
    for i, (img1path, _, flowpath) in enumerate(tqdm(test_list)):

        img1 = imread(img1path, pilmode='RGB')
        visual = img1[:,:,:3]

        if flowpath is not None:
            flo = load_flo(flowpath,'float32')
            rgbflo = flow2rgb(flo)
            visual = np.hstack((visual, rgbflo))

        if args.pred_dir is not None:
            fpath = img1path.replace(args.data, args.pred_dir)
            fpath = fpath.replace('.png', '.flo')
            fpath = fpath.replace('/composition/', '/')
            if os.path.isfile(fpath):
                predflow = flow2rgb(load_flo(fpath,'float16'))
            # Simple upsample for now so that the images are the same size
            resized = skimage.transform.resize(predflow, img1.shape, preserve_range=True)
            visual = np.hstack((visual, resized))
        
        save_path = img1path.replace(args.data, args.output_dir).replace('/composition/', '/')
        os.system('mkdir -p '+os.path.dirname(save_path))
        imsave(save_path, visual)


def flow2rgb(flow_map, max_value=None):
    flow_map_np = flow_map.transpose(2,0,1)
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    rgb_flow = rgb_map.clip(0,1)
    rgb_flow = (rgb_flow * 255).astype(np.uint8).transpose(1,2,0)
    return rgb_flow


def load_flo(path,type):
    with open(path, 'rb') as f:
        # head = np.fromfile(f, dtype='|S4',count=1).astype('|U4')
        # assert((head == 'PIEH').all()),'Header CHAR incorrect. Invalid .flo file'
        magic = np.fromfile(f, np.float32, count=1)
        assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        if type == 'float16':
            data = np.fromfile(f, np.float16, count=2*w*h)
        else:
            data = np.fromfile(f, np.float32, count=2*w*h)

    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (w, h, 2))
    return data2D


def make_real_dataset(dir, phase='test'):
    '''Will search for triplets that go by the pattern '[name]_img1.ppm  [name]_img2.ppm    [name]_flow.flo' '''
    images = []
    print('Fetching images from', glob.glob(os.path.join(dir, phase+'/*/composition/*.png')))
    for img1 in sorted( glob.glob(os.path.join(dir, phase+'/*/composition/*.png')) ):
        img2 = img1[:-9] + str(int(img1.split('/')[-1][:-4])+1).zfill(5) + '.png'

        if int(img1.split('/')[-1][:-4]) % 10 == 9:
            continue

        if not (os.path.isfile(os.path.join(dir,img1)) and os.path.isfile(os.path.join(dir,img2))):
            continue

        flow = img1.replace('.png', '.flo').replace('/composition/', '/flow/')
        if os.path.isfile(flow):
            images.append([img1, img2, flow])
        else:
            images.append([img1, img2, None])

    return images 


if __name__ == '__main__':
    main()
