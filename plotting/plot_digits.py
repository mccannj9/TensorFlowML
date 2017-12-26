#! /usr/bin/env python3

import numpy as np


def compute_prediction(image, weights):
    import numpy as np

    if weights == None:
        return 0

    image = np.asarray(image)/255
    hidden_bias = np.loadtxt(weights[0])
    hidden_kernel = np.loadtxt(weights[1])
    pred_bias = np.loadtxt(weights[2])
    pred_kernel = np.loadtxt(weights[3])

    # FeedForward predictions
    z2 = np.matmul(image, hidden_kernel) + hidden_bias
    a2 = 1/(1+np.exp(z2))
    z3 = np.matmul(a2, pred_kernel) + pred_bias
    a3 = 1/(1+np.exp(z3))
    value = np.argmax(a3)

    return (list(a3), value)


def ascii_plot(image_vector, nrows, ncols):

    print()
    im = image_vector
    im_bool = [str(int(not(not(x)))) for x in im]
    im_mat = [im_bool[x*nrows:x*nrows+nrows] for x in range(nrows)]
    output_matrix = ["".join(im_mat[i]) for i in range(len(im_mat))]
    output = "\n".join(output_matrix)
    print(output)
    return output


def bokeh_plot(images, label, image_number, nrows=28, ncols=28):

    from bokeh.plotting import figure, show, output_file, save

    im = np.flip(np.asarray(images).reshape(nrows,ncols), axis=0)
    img = np.empty((nrows,ncols), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape(nrows, ncols, 4)

    for i in range(nrows):
        for j in range(ncols):
            view[i, j, 0] = im[i,j]
            view[i, j, 1] = im[i,j]
            view[i, j, 2] = im[i,j]
            view[i, j, 3] = 255

    p = figure(x_range=(0,10), y_range=(0,10))

    # must give a vector of images
    p.image_rgba(image=[img], x=0, y=0, dw=10, dh=10)
    output_file("/mnt/Data/GitHub/TensorFlowML/data/image_rgba_%s.html" % image_number, title="image_rgba.py example")
    save(p, filename="/mnt/Data/GitHub/TensorFlowML/data/image_rgba_%s.html" % image_number, title="Image %s -- Handwritten %s" % (image_number, label))



def plot_and_predict(images_file, labels_file, weights, image_number=1, nrows=28, ncols=28):

    images = open(images_file)
    labels = open(labels_file)

    counter = 0
    for im, ll in zip(images, labels):
        counter += 1
        if counter == image_number:
            break

    im = [int(x) for x in im.split()]
    im = np.asarray(im)
    ll = ll.rstrip()

    output = ascii_plot(im, nrows, ncols)
    bokeh_plot(im, ll, image_number, nrows, ncols)

    print("This image is labeled as: %s" % ll)
    probs, label = compute_prediction(im, weights)
    print("This image is predicted to be: %s" % label)
    probs = ["{0:.5f}".format(x) for x in probs]
    probs = "  ".join(probs)
    print(probs)


def main():

    import argparse

    parser = argparse.ArgumentParser(description='Plot MNIST images and predictions')
    parser.add_argument('-i', '--images', type=str, help="images file", required=True)
    parser.add_argument('-l', '--labels', type=str, help="labels file", required=True)
    parser.add_argument('-n', '--image_number', type=int, help="line number of image", required=False, default=1)
    parser.add_argument('-w', '--weights', nargs="+", type=str, help="weights for predictions", required=False, default=None)

    args = parser.parse_args()

    plot_and_predict(args.images, args.labels, args.weights, image_number=args.image_number)

if __name__ == '__main__':
    main()
