import cv2
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import img_utils22 as imu

def main():
    pipe = imu.Pipe() | imu.LoadFile('./out/3003_test.png') \
        | imu.ExtractObjects(bgcolor=255) | imu.ShowImage('source')
    img1 = pipe(None)

    pipe = imu.Pipe() | imu.LoadFile('./images/3003/3003_2030.png') \
        | imu.ExtractObjects(bgcolor=237) | imu.ShowImage('best match')
    img2 = pipe(None)

    aligned = imu.align_images(img1, img2, mode='euclidian', pad_color=(237, 237, 237))
    imu.ShowImage()(aligned, 'align')

    score, diff = imu.get_image_diff(img1, aligned, align_mode=None, multichannel=False)
    print(score)
    imu.ShowImage()(diff, 'diff')
    return

    files = []
    for d in ['3003', '3070', '3004']:
        files.extend([str(f) for f in Path(f'./images/{d}/').glob('*.png')])
    print(f'{len(files)} files loaded')

    scores = []
    pipe2 = imu.Pipe() | imu.LoadFile(None) | imu.ExtractObjects(bgcolor=237)

    # files_subset = random.sample(files, 10)
    files_subset = files
    for f in (pbar := tqdm(files_subset)):
        pbar.set_description(Path(f).name)
        pipe2[imu.LoadFile].filename = f
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

    pipe2[imu.LoadFile].filename = fn
    pipe2 |= imu.ShowImage(Path(fn).name)
    pipe2(None)

if __name__ == '__main__':
    try:
        main()
    finally:
        cv2.destroyAllWindows()
