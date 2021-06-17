import cv2
import numpy as np
from scipy.ndimage.measurements import label
from scipy.ndimage.filters import gaussian_filter

def post_process(image: np.ndarray, seg_mask: str, blur: float = 0, resized: int = 160) -> np.ndarray:
    seg_mask = cv2.resize(
            np.array(
                cv2.imread(seg_mask, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
            ).astype(np.uint8),
        (resized,resized))
    image = np.moveaxis(image, 0, -1)
    image = cv2.resize(image.astype(float), (resized, resized))
    output = image.copy()

    background = np.median(image[seg_mask == 0], axis=0)
    output[seg_mask == 0] = background
    

    if blur != 0:
        for i in range(1,53):
            labeled_filter, num = label((seg_mask == i).astype(np.uint8), output=np.uint8)
            for j in range(1, num+1):
                mask = ((labeled_filter == j).astype(np.float))

                filter0 = gaussian_filter(image[:,:,0] * mask, sigma=blur)
                filter1 = gaussian_filter(image[:,:,1] * mask, sigma=blur)
                #weights = gaussian_filter(mask, sigma=blur)
                #filter0 /= weights
                #filter1 /= weights

                output[mask == 1] = [0,0]

                output[:, :, 0] += np.nan_to_num(filter0)
                output[:, :, 1] += np.nan_to_num(filter1)
    
    
    output = cv2.resize(output, (160, 160))
    return np.moveaxis(output, -1, 0)
