import cv2
import imutils
import numpy as np


def global_denoise(img, ksize=3):
    return cv2.medianBlur(img, ksize)


def _add_random_noise(img, bright_noise=0, dark_noise=0):
    """
    Don't use this method in pre-processing, for this method is to raise the difficulty for algorithm testing
    :param bright_noise: the quantity of bright_noise
    :param dark_noise: the quantity of dark_noise
    :return:
    """
    img_bk = img.copy()

    h, w = img.shape[:2]

    if len(img.shape) == 3:
        c = img.shape[2]
        bright_size = (1, bright_noise, c)
        dark_size = (1, dark_noise, c)
    else:
        bright_size = (1, bright_noise)
        dark_size = (1, dark_noise)

    bright_noise_x = np.random.randint(0, w, size=(1, bright_noise))
    bright = np.random.randint(0, h, size=(1, bright_noise))
    img_bk[bright, bright_noise_x] = np.random.randint(192, 256, size=bright_size)

    dark_noise_x = np.random.randint(0, w, size=(1, dark_noise))
    dark_noise_y = np.random.randint(0, h, size=(1, dark_noise))
    img_bk[dark_noise_y, dark_noise_x] = np.random.randint(0, 64, size=dark_size)

    return img_bk


if __name__ == "__main__":
    img = cv2.imread("test.jpg")
    img = imutils.resize(img, width=800)

    # noise = _add_random_noise(img, 200, 20000)
    # cv2.imshow("Noise", noise)
    # cv2.waitKey(0)

    global_denoi = global_denoise(img, 3)

    delta = img.astype(np.int16) - global_denoi.astype(np.int16)
    # _, delta = cv2.threshold(delta, 240, 255, cv2.THRESH_BINARY)
    delta = cv2.convertScaleAbs(delta)
    cv2.imshow("Delta", delta)
    cv2.waitKey(0)

    patch_size = 3
    a = np.where(delta > 0)
    print(0)