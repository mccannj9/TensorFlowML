#! /usr/bin/env python3

def ascii_plot(images_file, labels_file, weights, image_number=1, nrows=28, ncols=28):

    print()
    images = open(images_file)
    labels = open(labels_file)

    counter = 0
    for im, ll in zip(images, labels):
        counter += 1
        if counter == image_number:
            break

    im = [int(x) for x in im.split()]
    im_bool = [str(int(not(not(x)))) for x in im]
    ll = ll.rstrip()

    im_mat = [im_bool[x*nrows:x*nrows+nrows] for x in range(nrows)]
    output_matrix = ["".join(im_mat[i]) for i in range(len(im_mat))]
    output = "\n".join(output_matrix)
    print(output)
    print("This image is labeled as: %s" % ll)
    probs, label = compute_prediction(im, weights)
    print("This image is predicted to be: %s" % label)
    print([round(x, 3) for x in probs])


def bokeh_plot(images_file, labels_file, image_number=1, nrows=28, ncols=28):
    pass


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


def main():

    import argparse

    parser = argparse.ArgumentParser(description='Convert MNIST to TXT')
    parser.add_argument('-i', '--images', type=str, help="images file", required=True)
    parser.add_argument('-l', '--labels', type=str, help="labels file", required=True)
    parser.add_argument('-n', '--image_number', type=int, help="line number of image", required=False, default=1)
    parser.add_argument('-w', '--weights', nargs="+", type=str, help="weights for predictions", required=False, default=None)

    args = parser.parse_args()

    ascii_plot(args.images, args.labels, args.weights, image_number=args.image_number)
    # print(args.weights)

if __name__ == '__main__':
    main()
