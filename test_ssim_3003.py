# LEGO sorter project
# Experiments with SSIM-based image classifications
# (c) kol, 2022
#
# Actually these are not very successfull as images are quite different and cant be even properly aligned
# Max.SSIM score achieved for test image is 0.53, which is quite below expected level

import cv2
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import img_utils22 as imu

IMAGE_DIR = str(Path(__file__).parent.joinpath('images'))

def get_all_images():
    return Path(IMAGE_DIR).rglob('*.png')

def main():
    pipe = imu.Pipe() | imu.LoadFile('./out/3003_test.png') \
        | imu.ExtractObjects(bgcolor=255) | imu.ShowImage('source')
    img1 = pipe(None)

    files = get_all_images()
    scores = []
    pipe2 = imu.Pipe() | imu.LoadFile(None) | imu.ExtractObjects(bgcolor=237)

    files_subset = random.sample(list(files), 10)
    # files_subset = files
    for f in (pbar := tqdm(files_subset)):
        pbar.set_description(Path(f).name)
        pipe2[imu.LoadFile].filename = str(f)
        img2 = pipe2(None)
        try:
            score, _ = imu.get_image_diff(img1, img2, align_mode='euclidian', multichannel=False, 
                pad_color=imu.COLOR_WHITE)
            scores.append(score)
        except:
            scores.append(0)

    max_score_idx = np.argmax(scores)
    fn, score = files_subset[max_score_idx], scores[max_score_idx]
    print(f'Max similarity score {score} detected on {fn}')

    pipe2[imu.LoadFile].filename = str(fn)
    pipe2 |= imu.ShowImage(Path(fn).name)
    img3 = pipe2(None)
    _, diff = imu.get_image_diff(img1, img3, align_mode='euclidian', multichannel=False, 
                pad_color=imu.COLOR_WHITE)
    imu.ShowImage()(diff, 'diff')

if __name__ == '__main__':
    try:
        main()
    finally:
        cv2.destroyAllWindows()
