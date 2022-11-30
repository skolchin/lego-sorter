import cv2
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import img_utils22 as imu

def imshow(img, title='debug'):
    if img is not None:
        cv2.imshow(title, img)
        cv2.waitKey(0)

def main():
    pipe = imu.Pipe() | imu.LoadFile('./out/3003_test.png') | imu.Resize((256,256))
    img1 = pipe(None)
    imshow(img1, 'image1')

    img1 = imu.get_image_area(img1, (50, 50, 200, 200))

    files = []
    for d in ['3003', '3070', '3700']:
        files.extend([str(f) for f in Path(f'./images/{d}/').glob('*.png')])
    print(f'{len(files)} files loaded')

    scores = []
    pipe2 = imu.Pipe() | imu.LoadFile(None) | imu.Resize((256,256)) | imu.Area((50,70,200,220))

    # file_subset = random.sample(files, 10)
    file_subset = files

    for f in (pbar := tqdm(file_subset)):
        pbar.set_description(Path(f).name)
        pipe2[imu.LoadFile].filename = f
        img2 = pipe2(None)
        # imshow(img2, f)
        try:
            score, _ = imu.get_image_diff(img1, img2, align_mode='euclidian', multichannel=False, 
                pad_color=imu.COLOR_WHITE)
            scores.append(score)
        except:
            scores.append(0)

    max_score_idx = np.argmax(scores)
    fn, score = files[max_score_idx], scores[max_score_idx]
    print(f'Max similarity score {score} is for {fn}')

    pipe[imu.LoadFile].filename = fn
    max_score_img = pipe(None)
    imshow(max_score_img, Path(fn).name)

if __name__ == '__main__':
    try:
        main()
    finally:
        cv2.destroyAllWindows()