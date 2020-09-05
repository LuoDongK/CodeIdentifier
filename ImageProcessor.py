import cv2 as cv
import glob
import os
import random
import csv


def random_rotation(img, num, label, file):
    (h, w) = img.shape
    center = (w // 2, h // 2)
    for ii in range(1, num + 1):
        mat = cv.getRotationMatrix2D(center, random.uniform(-3, 3), 1.1)
        rotated = cv.warpAffine(img, mat, (w, h))
        _, thresh = cv.threshold(rotated, 50, 255, cv.THRESH_BINARY)
        row = list(thresh.reshape(-1))
        row.append(label)
        with open(file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


def process(origin_root, file):
    images = []
    images += glob.glob(os.path.join(origin_root, '*.jpg'))
    for index, image in enumerate(images):
        print(index)
        if index >= 40000:
            file = 'validate.csv'
        label = image.split(os.sep)[-1][:-4]
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        canny = cv.Canny(gray, 50, 200)
        stats = cv.connectedComponentsWithStats(canny)[2]
        for stat in stats:
            if stat[4] < 20:
                cv.rectangle(canny, tuple(stat[0:2]), tuple(stat[0:2] + stat[2:4]), 0, thickness=-1)

        row = list(canny.reshape(-1))
        row.append(label)
        # random_rotation(canny, 2, label, file)
        with open(file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


if __name__ == '__main__':
    process('image', 'train.csv')
