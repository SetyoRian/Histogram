import cv2
import numpy as np
from matplotlib import pyplot as plt


def gray():
    gray_img = cv2.imread('aurora.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Gray Image', gray_img)
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    plt.hist(gray_img.ravel(), 256, [0, 256])
    plt.title('Histogram for gray scale picture')
    plt.show()

    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break  # ESC key to exit
    cv2.destroyAllWindows()


def rgb():
    img = cv2.imread('aurora.jpg', -1)
    cv2.imshow('RGB Image', img)

    color = ('b', 'g', 'r')
    for channel, col in enumerate(color):
        histr = cv2.calcHist([img], [channel], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.title('Histogram for color scale picture')
    plt.show()

    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break  # ESC key to exit
    cv2.destroyAllWindows()


def strech():
    img = cv2.imread('aurora.jpg')
    original = img.copy()
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    img = cv2.LUT(img, table)
    cv2.imshow("Before", original)
    cv2.imshow("After", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def equ():
    img = cv2.imread('aurora.jpg', 0)
    equ = cv2.equalizeHist(img)
    result = np.hstack((img, equ))
    plt.plot(result)
    cv2.imshow("Equalization", result)
    plt.show()
    cv2.imwrite('result.png', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    while 1:
        print("1. Histogram for Grayscale Image")
        print("2. Histogram for RGB Image")
        print("3. Stretching Image")
        print("4. Equalization Image")
        print("Choose : ")
        ch = int(input())
        if ch == 1:
            gray()
        elif ch == 2:
            rgb()
        elif ch == 3:
            strech()
        elif ch == 4:
            equ()
        else:
            print("Invalid Option")


main()
