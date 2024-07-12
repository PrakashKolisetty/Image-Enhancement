import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import math


def SaltAndPaper(image, density):
    # create an empty array with same size as input image
    output = np.zeros(image.shape, np.uint8)

    # parameter for controlling how much salt and paper are added
    threshhold = 1 - density

    # loop every each pixel and decide add the noise or not base on threshhold (density)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            possibility = random.random()
            if possibility < density:
                output[i][j] = 0
            elif possibility > threshhold:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def MeanFilter(image, filter_size):
    # create an empty array with same size as input image
    output = np.zeros(image.shape, np.uint8)

    # creat an empty variable
    result = 0

    # deal with filter size = 3x3
    if filter_size == 9:
        for j in range(1, image.shape[0]-1):
            for i in range(1, image.shape[1]-1):
                for y in range(-1, 2):
                    for x in range(-1, 2):
                        result = result + image[j+y, i+x]
                output[j][i] = int(result / filter_size)
                result = 0

    # deal with filter size = 5x5
    elif filter_size == 25:
        for j in range(2, image.shape[0]-2):
            for i in range(2, image.shape[1]-2):
                for y in range(-2, 3):
                    for x in range(-2, 3):
                        result = result + image[j+y, i+x]
                output[j][i] = int(result / filter_size)
                result = 0

    return output


def MedianFilter(image, filter_size):
    # create an empty array with same size as input image
    output = np.zeros(image.shape, np.uint8)

    # create the kernel array of filter as same size as filter_size
    filter_array = [image[0][0]] * filter_size

    # deal with filter size = 3x3
    if filter_size == 9:
        for j in range(1, image.shape[0]-1):
            for i in range(1, image.shape[1]-1):
                filter_array[0] = image[j-1, i-1]
                filter_array[1] = image[j, i-1]
                filter_array[2] = image[j+1, i-1]
                filter_array[3] = image[j-1, i]
                filter_array[4] = image[j, i]
                filter_array[5] = image[j+1, i]
                filter_array[6] = image[j-1, i+1]
                filter_array[7] = image[j, i+1]
                filter_array[8] = image[j+1, i+1]

                # sort the array
                filter_array.sort()

                # put the median number into output array
                output[j][i] = filter_array[4]

    # deal with filter size = 5x5
    elif filter_size == 25:
        for j in range(2, image.shape[0]-2):
            for i in range(2, image.shape[1]-2):
                filter_array[0] = image[j-2, i-2]
                filter_array[1] = image[j-1, i-2]
                filter_array[2] = image[j, i-2]
                filter_array[3] = image[j+1, i-2]
                filter_array[4] = image[j+2, i-2]
                filter_array[5] = image[j-2, i-1]
                filter_array[6] = image[j-1, i-1]
                filter_array[7] = image[j, i-1]
                filter_array[8] = image[j+1, i-1]
                filter_array[9] = image[j+2, i-1]
                filter_array[10] = image[j-2, i]
                filter_array[11] = image[j-1, i]
                filter_array[12] = image[j, i]
                filter_array[13] = image[j+1, i]
                filter_array[14] = image[j+2, i]
                filter_array[15] = image[j-2, i+1]
                filter_array[16] = image[j-1, i+1]
                filter_array[17] = image[j, i+1]
                filter_array[18] = image[j+1, i+1]
                filter_array[19] = image[j+2, i+1]
                filter_array[20] = image[j-2, i+2]
                filter_array[21] = image[j-1, i+2]
                filter_array[22] = image[j, i+2]
                filter_array[23] = image[j+1, i+2]
                filter_array[24] = image[j+2, i+2]

                # sort the array
                filter_array.sort()

                # put the median number into output array
                output[j][i] = filter_array[12]
    return output


def main():
    # read image
    gray_lena = cv2.imread("N:/nithe/databasse/imageprocessing/eagle.jpg", 0)

    # add salt and paper (0.01 is a proper parameter)"N:\nithe\databasse\imageprocessing\up1.jpg"
    noise_lena = SaltAndPaper(gray_lena, 0.01)

    # use 3x3 mean filter
    mean_3x3_lena = MeanFilter(noise_lena, 9)

    # use 3x3 median filter
    median_3x3_lena = MedianFilter(noise_lena, 9)

    # use 5x5 mean filter
    mean_5x5_lena = MeanFilter(noise_lena, 25)

    # use 5x5 median filter
    median_5x5_lena = MedianFilter(noise_lena, 25)
    cv2.imshow('Normal',gray_lena)
    cv2.waitKey()
    # cv2.imwrite('N:/nithe/databasse/imageprocessing/sample.jpg',gray_lena )

    # display the salt and paper image


    cv2.imshow('Salt and peper',noise_lena)
    cv2.waitKey()
    # cv2.imwrite('N:/nithe/databasse/imageprocessing/salt_paper.jpg', noise_lena)

    dst = cv2.equalizeHist(noise_lena)
    #cv2.imwrite('N:/nithe/databasse/imageprocessing/saltPaperHE.jpg', dst)
    cv2.imshow('Hist', dst)
    cv2.waitKey()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(noise_lena)
    cv2.imshow("CLAHE", cl1)
    cv2.waitKey()
    # cv2.imwrite('N:/nithe/databasse/imageprocessing/saltPaperCLAHE.jpg', cl1)

    # display 3x3 mean filter
    cv2.imshow('Mean',mean_3x3_lena,)
    cv2.waitKey()
    #cv2.imwrite('N:/nithe/databasse/imageprocessing/3x3mean.jpg', mean_3x3_lena)
    # image2 = cv2.cvtColor(mean_3x3_lena, cv2.COLOR_BGR2GRAY)
    dst2 = cv2.equalizeHist(mean_3x3_lena)
    cv2.imshow('Hist', dst2)
    cv2.waitKey()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl2 = clahe.apply(mean_3x3_lena)
    cv2.imshow("Mean Filter CLAHE", cl2)
    cv2.waitKey()


    # display 3x3 median filter


    cv2.imshow('Median',median_3x3_lena)
    cv2.waitKey()
    #cv2.imwrite('N:/nithe/databasse/imageprocessing/3x3median.jpg', median_3x3_lena)
    # image3 = cv2.cvtColor(median_3x3_lena, cv2.COLOR_BGR2GRAY)
    dst3 = cv2.equalizeHist(median_3x3_lena)
    cv2.imshow('Hist', dst3)
    cv2.waitKey()
    # cv2.imwrite('N:/nithe/databasse/imageprocessing/median3x3HE.jpg', dst3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl3 = clahe.apply(median_3x3_lena)
    cv2.imshow("median", cl3)
    cv2.waitKey()
    # cv2.imwrite('N:/nithe/databasse/imageprocessing/median3x3CLAHE.jpg', cl3)
    cv2.imwrite('N:/nithe/databasse/imageprocessing/up1CLAHE.jpg', cl3)
    # display 5x5 median filter
    cv2.imwrite('N:/nithe/databasse/imageprocessing/up1eq.jpg', median_3x3_lena)
    cv2.imshow('Mean',mean_5x5_lena)
    cv2.waitKey()
    #cv2.imwrite('N:/nithe/databasse/imageprocessing/5x5mean.jpg', mean_5x5_lena)
    # image4 = cv2.cvtColor(mean_5x5_lena, cv2.COLOR_BGR2GRAY)
    dst4 = cv2.equalizeHist(mean_5x5_lena)
    # cv2.imwrite('N:/nithe/databasse/imageprocessing/mean5x5HE.jpg', dst4)
    cv2.imshow('Hist', dst4)
    cv2.waitKey()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl4 = clahe.apply(mean_5x5_lena)
    cv2.imshow("Mean", cl4)
    cv2.waitKey()
    # cv2.imwrite('N:/nithe/databasse/imageprocessing/mean5x5CLAHE.jpg', cl4)

    # display 5x5 median filter

    cv2.imshow('Median 5',median_5x5_lena)
    cv2.waitKey()

    #cv2.imwrite('N:/nithe/databasse/imageprocessing/5x5median.jpg', median_5x5_lena)
    # image5 = cv2.cvtColor(median_5x5_lena, cv2.COLOR_BGR2GRAY)
    dst5 = cv2.equalizeHist(median_5x5_lena)
    cv2.imshow('Hist', dst5)
    cv2.waitKey()
    # cv2.imwrite('N:/nithe/databasse/imageprocessing/median5x5HE.jpg', dst5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl5 = clahe.apply(median_5x5_lena)
    cv2.imshow('CLAHE', cl5)
    cv2.waitKey()
    # cv2.imwrite('N:/nithe/databasse/imageprocessing/median5x5CLAHE.jpg', cl5)




# if __name__ == "__main__":
main()
