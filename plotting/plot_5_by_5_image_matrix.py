#! /usr/bin/env python3

import numpy as np

def build_rgb_tensor(image, nrows=28, ncols=28):

    # converts pixel list to pixel matrix image representation
    im = np.flip(np.asarray(images).reshape(nrows,ncols), axis=0)
    # tensor for rgb grayscale representation
    img = np.empty((nrows,ncols), dtype=np.uint32)
    # reshaping done in place? is variable necessary?
    view = img.view(dtype=np.uint8).reshape(nrows, ncols, 4)
    # lookup how to pad edges of matrix on stack exchange!!!
    for i in range(nrows):
        for j in range(ncols):
            # always the same rgb values for grayscale
            view[i, j, 0] = im[i,j]
            view[i, j, 1] = im[i,j]
            view[i, j, 2] = im[i,j]
            # no opacity
            view[i, j, 3] = 255

    return img


def get_random_images(image_matrix, nimages=25):
    return np.random.permutation(image_matrix.shape[0])[:nimages]


def concatenate_image_tensors(tensors):
    # some numpy code here for concatenation
    # need to make 5 by 5 matrix
    tensor = None
    return tensor


def bokeh_plot_matrix(images, nrows=28, ncols=28):

    from bokeh.plotting import figure, show, output_file, save

    images = get_random_images(images)
    rgb_tensors = [build_rgb_tensor(im) for im in images]

    img = concatenate_image_tensors(rgb_tensors)
    p = figure(x_range=(0,10), y_range=(0,10))

    # must give a vector of images
    p.image_rgba(image=[img], x=0, y=0, dw=10, dh=10)
    output_file("/mnt/Data/GitHub/TensorFlowML/data/image_rgba_%s.html" % image_number, title="MNIST Image Sample")
    save(p, filename="/mnt/Data/GitHub/TensorFlowML/data/image_rgba_%s.html" % image_number, title="MNIST Image Sample")

    return 0


def main():

    import argparse

    parser = argparse.ArgumentParser(description='Plot MNIST images and predictions')
    parser.add_argument('-i', '--images', type=str, help="images file", required=True)
    parser.add_argument('-o', '--output', type=str, help="html output file", required=True)

    args = parser.parse_args()

    return 0


if __name__ == '__main__':
    main()
