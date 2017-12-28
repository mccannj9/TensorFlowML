#! /usr/bin/env python3

import numpy as np

def build_rgb_tensor(image, nrows=28, ncols=28):

    # converts pixel list to pixel matrix image representation
    im = np.flip(np.asarray(image).reshape(nrows,ncols), axis=0)
    im = np.pad(im, 1, 'constant', constant_values=255)
    # print(im.shape)
    ncols += 2
    nrows += 2
    # tensor for rgb grayscale representation
    img = np.empty((nrows,ncols), dtype=np.uint32)
    # reshaping done in place? is variable necessary?
    view = img.view(dtype=np.uint8).reshape(nrows, ncols, 4)
    # +1 accounts for padding
    for i in range(nrows):
        for j in range(ncols):
            # always the same rgb values for grayscale
            view[i, j, 0] = im[i,j]
            view[i, j, 1] = im[i,j]
            view[i, j, 2] = im[i,j]
            # no opacity
            view[i, j, 3] = 255

    # print("image shape")
    # print(img.shape)

    return img


def get_random_images(image_matrix, nimages=25):
    indices = np.random.permutation(image_matrix.shape[0])[:nimages]
    images = image_matrix[indices]
    return images


def concatenate_image_tensors(tensors):
    # some numpy code here for concatenation
    # need to make 5 by 5 matrix
    # print(len(tensors))
    sublists = [tensors[x:x+5] for x in range(0, len(tensors), 5)]
    # print(len(sublists))

    concatenated = []
    for sublist in sublists:
        init_tensor = sublist[0]
        for tensor in sublist[1:]:
            init_tensor = np.concatenate((init_tensor, tensor), axis=1)
        concatenated.append(init_tensor)
        # print(init_tensor.shape)

    init_tensor = concatenated[0]
    for tensor in concatenated[1:]:
        init_tensor = np.concatenate((init_tensor,tensor), axis=0)

    return init_tensor


def bokeh_plot_matrix(images, output, nrows=28, ncols=28):

    from bokeh.plotting import figure, show, output_file, save

    images = get_random_images(images)
    rgb_tensors = [build_rgb_tensor(im) for im in images]

    # print(len(rgb_tensors))
    img = concatenate_image_tensors(rgb_tensors)
    p = figure(x_range=(0,30), y_range=(0,30))

    # must give a vector of images
    p.image_rgba(image=[img], x=0, y=0, dw=30, dh=30)
    output_file(output, title="MNIST Image Sample")
    save(p, filename=output, title="MNIST Image Sample")

    return 0


def main():

    import argparse

    parser = argparse.ArgumentParser(description='Plot MNIST images and predictions')
    parser.add_argument('-i', '--images', type=str, help="images file", required=True)
    parser.add_argument('-o', '--output', type=str, help="html output file", required=True)

    args = parser.parse_args()

    image_data = np.loadtxt(args.images)
    bokeh_plot_matrix(image_data, args.output)

    return 0


if __name__ == '__main__':
    main()
