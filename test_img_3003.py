import cv2
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import img_utils22 as imu

def main():
    pipe = imu.Pipe() | imu.LoadFile('./out/3003_test.png') \
        | imu.ExtractObjects(bgcolor=255) | imu.ShowImage('image1') #| imu.Resize((256,256))
    img1 = pipe(None)

    # pipe = imu.Pipe() | imu.LoadFile('./images/3003/3003_1840.png') \
    #     | imu.EqualizeLuminosity() | imu.ExtractObjects(bgcolor=237) #| imu.Resize((256,256))
    # img2 = pipe(None)
    # imshow(img2, 'image2')

    # return

    files = []
    for d in ['3003', '3070', '3004']:
        files.extend([str(f) for f in Path(f'./images/{d}/').glob('*.png')])
    print(f'{len(files)} files loaded')

    scores = []
    pipe2 = imu.Pipe() | imu.LoadFile(None) | imu.ExtractObjects(bgcolor=237) #| imu.Resize((256,256))

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
    fn, score = file_subset[max_score_idx], scores[max_score_idx]
    print(f'Max similarity score {score} detected on {fn}')

    pipe[imu.LoadFile].filename = fn
    max_score_img = pipe(None)
    imu.ShowImage()(max_score_img, Path(fn).name)

if __name__ == '__main__':
    try:
        main()
    finally:
        cv2.destroyAllWindows()
