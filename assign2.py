################################################################################
# COMP3317 Computer Vision
# Assignment 2 - Conrner detection
################################################################################
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
from pathlib import Path

################################################################################
#  perform RGB to grayscale conversion
################################################################################
def rgb2gray(img_color) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    # return:
    #    img_gray - a h x w numpy ndarray (dtype = np.float64) holding
    #               the grayscale image

    # TODO: using the Y channel of the YIQ model to perform the conversion
    # I(i,j) = 0.299*R(i, j) + 0.587*G(i, j) + 0.114*B(i, j)

    i = len(img_color)
    j = len(img_color[1])
    img_gray = np.zeros((i,j))
    for x in range(0,i):
        for y in range(0,j):
            img_gray[x][y] = 0.299*img_color[x][y][0] + 0.587*img_color[x][y][1] + 0.114*img_color[x][y][2]
    return img_gray

################################################################################
#  perform 1D smoothing using a 1D horizontal Gaussian filter
################################################################################
def smooth1D(img, sigma) :
    # input :
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the 1D Gaussian function
    # return:
    #    img_smoothed - a h x w numpy ndarry holding the 1D smoothing result

    n = 3*sigma

    x = np.arange(-1*n, n+1)
    filterz = np.exp((x ** 2)/-2/(sigma ** 2))
    peak_trunc = (1/1000) * np.exp(0)
    filter_trunc = np.extract(filterz >= peak_trunc, filterz)


    result = convolve1d(img, filter_trunc, 1, np.float64, 'constant', 0, 0)

    i = len(result)
    j = len(result[1])
    m_ones = np.ones((i,j))

    weight = convolve1d(m_ones, filter_trunc, 1, np.float64, 'constant', 0, 0)

    img_smoothed = np.divide(result, weight)

    return img_smoothed

################################################################################
#  perform 2D smoothing using 1D convolutions
################################################################################
def smooth2D(img, sigma) :
    # input:
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the Gaussian function
    # return:
    #    img_smoothed - a h x w numpy array holding the 2D smoothing result
    hor_smooth = smooth1D(img, sigma)
    ver_smooth = smooth1D(hor_smooth.T, sigma)
    img_smoothed = ver_smooth.T

    return img_smoothed

################################################################################
#   perform Harris corner detection
################################################################################
def harris(img, sigma, threshold) :
    # input:
    #    img - a h x w numpy ndarry holding the input image
    #    sigma - sigma value of the Gaussian function used in smoothing
    #    threshold - threshold value used for identifying corners
    # return:
    #    corners - a list of tuples (x, y, r) specifying the coordinates
    #              (up to sub-pixel accuracy) and cornerness value of each corner

    filter = np.array([0.5, 0, -0.5])
    Ix = convolve1d(img, filter, 1, np.float64, 'constant', 0, 0)
    Iy = (convolve1d(img.T, filter, 1, np.float64, 'constant', 0, 0)).T

    Ix2 = np.square(Ix)
    Iy2 = np.square(Iy)
    IxIy = np.multiply(Ix,Iy)


    Ix2_smooth = smooth2D(Ix2, sigma)
    Iy2_smooth = smooth2D(Iy2, sigma)
    IxIy_smooth = smooth2D(IxIy, sigma)

    i = len(img)
    j = len(img[0])

    r = np.zeros((i,j))
    corners = []

    r = (Ix2_smooth*Iy2_smooth - IxIy_smooth*IxIy_smooth) - (0.04*(Ix2_smooth+Iy2_smooth)**2)

    corners = []

    for x in range(1,i-1):
        for y in range(1,j-1):
            arr = np.array([r[x-1,y], r[x+1,y], r[x,y], r[x,y-1], r[x,y+1],
            r[x-1, y-1], r[x-1,y+1], r[x+1, y+1], r[x+1, y-1]])
            if r[x,y] == np.amax(arr):
                a = (r[x-1,y] + r[x+1,y] - 2*r[x,y])/2
                b = (r[x,y-1] + r[x,y+1] - 2*r[x,y])/2
                c = (r[x+1,y] - r[x-1,y])/2
                d = (r[x,y+1] - r[x,y-1])/2
                e = r[x,y]

                if(r[x,y] > threshold):
                    if (a != 0) and (b != 0):
                        x_sub = x-c/(2*a)
                        y_sub = y-d/(2*b)
                        corners.append([y_sub, x_sub, r[x,y]])

    return sorted(corners, key = lambda corner : corner[2], reverse = True)

################################################################################
#   save corners to a file
################################################################################
def save(outputfile, corners) :
    try :
        file = open(outputfile, 'w')
        file.write('%d\n' % len(corners))
        for corner in corners :
            file.write('%.4f %.4f %.4f\n' % corner)
        file.close()
    except :
        print('Error occurs in writting output to \'%s\''  % outputfile)
        sys.exit(1)

################################################################################
#   load corners from a file
################################################################################
def load(inputfile) :
    try :
        file = open(inputfile, 'r')
        line = file.readline()
        nc = int(line.strip())
        print('loading %d corners' % nc)
        corners = list()
        for i in range(nc) :
            line = file.readline()
            (x, y, r) = line.split()
            corners.append((float(x), float(y), float(r)))
        file.close()
        return corners
    except :
        print('Error occurs in writting output to \'%s\''  % outputfile)
        sys.exit(1)

################################################################################
## main
################################################################################
def main() :
    parser = argparse.ArgumentParser(description = 'COMP3317 Assignment 2')
    parser.add_argument('-i', '--inputfile', type = str, default = 'grid1.jpg', help = 'filename of input image')
    parser.add_argument('-s', '--sigma', type = float, default = 1.0, help = 'sigma value for Gaussain filter')
    parser.add_argument('-t', '--threshold', type = float, default = 1e6, help = 'threshold value for corner detection')
    parser.add_argument('-o', '--outputfile', type = str, help = 'filename for outputting corner detection result')
    args = parser.parse_args()

    print('------------------------------')
    print('COMP3317 Assignment 2')
    print('input file : %s' % args.inputfile)
    print('sigma      : %.2f' % args.sigma)
    print('threshold  : %.2e' % args.threshold)
    print('output file: %s' % args.outputfile)
    print('------------------------------')

    # load the image
    try :
        #img_color = imageio.imread(args.inputfile)

        img_color = plt.imread(args.inputfile)
        print('%s loaded...' % args.inputfile)
    except :
        print('Cannot open \'%s\'.' % args.inputfile)
        sys.exit(1)
    # uncomment the following 2 lines to show the color image
    # plt.imshow(np.uint8(img_color))
    # plt.show()

    # perform RGB to gray conversion
    print('perform RGB to grayscale conversion...')
    img_gray = rgb2gray(img_color)
    # uncomment the following 2 lines to show the grayscale image
    # plt.imshow(np.float32(im_smooth), cmap = 'gray')
    # plt.show()


    # perform corner detection
    print('perform Harris corner detection...')
    corners = harris(img_gray, args.sigma, args.threshold)
    #
    # # plot the corners
    #print(corners)
    print('%d corners detected...' % len(corners))
    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]
    fig = plt.figure()
    plt.imshow(np.float32(img_gray), cmap = 'gray')
    plt.plot(x, y,'r+',markersize = 5)
    plt.show()

    # save corners to a file
    if args.outputfile :
        save(args.outputfile, corners)
        print('corners saved to \'%s\'...' % args.outputfile)

if __name__ == '__main__':
    main()
