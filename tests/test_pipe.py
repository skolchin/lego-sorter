# LEGO sorter project
# Capture a frame for further analysys
# (c) kol, 2022

import os
import cv2
from absl import app
from root_dir import ROOT_DIR, OUTPUT_DIR

from lib.pipe_utils import *
from lib.image_dataset import _preprocess

def main(argv):
    ref = cv2.imread(os.path.join(ROOT_DIR, 'images', '3003', '3003_0.png'))
    ref = _preprocess(ref)[0].numpy().astype('uint8')
    imu.imshow(ref)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print('Cannot open camera, exiting')
        return

    cam.set(cv2.CAP_PROP_FPS, FPS_RATE)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[1])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[0])

    frame_count = 0
    show_hist = False

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = _preprocess(frame)[0].numpy().astype('uint8')
        if frame_count % FPS_RATE == 0 and show_hist:
            show_hist_window(frame, ref, (480, 640), log_scale=True)

        show_frame(frame)
        frame_count += 1

        key = int(cv2.waitKey(1) & 0xFF)
        match key:
            case 27 | 113:  # esc or q
                break

            case 99 | 32:    # c or space
                fn = os.path.join(ROOT_DIR, 'out', f'frame_{frame_count}.png')
                cv2.imwrite(fn, frame)
                print(f'Frame saved to {fn}')

            case 104:   # h
                if show_hist: hide_hist_window()
                show_hist = not show_hist

            case 115:   # s
                cam.set(cv2.CAP_PROP_SETTINGS, 1)

    cam.release()

if __name__ == '__main__':
    app.run(main)
