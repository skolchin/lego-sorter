import os
import cv2
from root_dir import ROOT_DIR

from lib.pipe_utils import *

def main():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print('Cannot open camera, exiting')
        return

    cam.set(cv2.CAP_PROP_FPS, FPS_RATE)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_SIZE[1])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_SIZE[0])

    frame_count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        show_stream(frame)
        frame_count += 1

        key = int(cv2.waitKey(1) & 0xFF)
        match key:
            case 27 | 113:  # esc or q
                break

            case 99 | 32:    # c or space
                fn = os.path.join(ROOT_DIR, 'out', f'frame_{frame_count}.png')
                cv2.imwrite(fn, frame)
                print(f'Frame saved to {fn}')

    cam.release()

if __name__ == '__main__':
    main()
