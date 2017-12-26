#! /usr/bin/env python3

def ascii_plot(images_file, labels_file, image_number=1, nrows=28, ncols=28):

    print()
    images = open(images_file)
    labels = open(labels_file)

    counter = 0
    for im, ll in zip(images, labels):
        counter += 1
        if counter == image_number:
            break

    im = [int(x) for x in im.split()]
    im = [str(int(not(not(x)))) for x in im]
    ll = ll.rstrip()

    im_mat = [im[x*nrows:x*nrows+nrows] for x in range(nrows)]
    output_matrix = ["".join(im_mat[i]) for i in range(len(im_mat))]
    output = "\n".join(output_matrix)
    print(output)
    print("This image is labeled as: %s" % ll)


def bokeh_plot(images_file, labels_file, image_number=1, nrows=28, ncols=28):
    pass


def main():

    import argparse

    parser = argparse.ArgumentParser(description='Convert MNIST to TXT')
    parser.add_argument('-i', '--images', type=str, help="images file", required=True)
    parser.add_argument('-l', '--labels', type=str, help="labels file", required=True)
    parser.add_argument('-n', '--image_number', type=int, help="line number of image", required=False, default=1)
    parser.add_argument('-w', '--weights', type=str, help="weights for predictions", required=False, default=None)


    args = parser.parse_args()

    ascii_plot(args.images, args.labels, image_number=args.image_number)

if __name__ == '__main__':
    main()
