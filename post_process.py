import cv2
import numpy as np


def post_process(image: np.ndarray, seg_mask: str) -> np.ndarray:
    seg_mask = np.array(cv2.imread(seg_mask, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]).astype(np.uint8)
    seg_mask_rs = cv2.resize(seg_mask, (160,160))
    image = np.moveaxis(image, [0,1,2], [2,1,0])
    output = image.copy()

    for i in range(1):
        background = image[seg_mask_rs == i].mean(axis=0)
        output[seg_mask_rs == i] = background
        
    return np.moveaxis(output, [0,1,2], [2,1,0])