# LEGO sorter project
# Pipeline tests - probably should be moved to img_utils22?
# (c) kol, 2022

import cv2
import numpy as np
import img_utils22 as imu

def main():
    pipe1 = imu.Pipe() | imu.LoadFile('./out/3003_test.png') | imu.ShowImage('source')
    img1 = pipe1(None)

    pipe2 = imu.Pipe() | imu.EqualizeLuminosity() | imu.Blur() | imu.Gray() | imu.Edges() | imu.Dilate(kernel_size=2) | imu.ShowImage('mask')
    mask = pipe2(img1)
    mask_inv = cv2.bitwise_not(mask)

    img2 = cv2.bitwise_and(img1, img1, mask=mask_inv)
    img_fg = np.full(img1.shape, imu.COLOR_BLACK, img1.dtype)
    img_fg = cv2.bitwise_and(img_fg, img_fg, mask=mask)
    img2 = cv2.add(img2, img_fg)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    imu.ShowImage('merged')(img2)

if __name__ == '__main__':
    try:
        main()
    finally:
        cv2.destroyAllWindows()
