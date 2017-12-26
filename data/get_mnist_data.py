#! /usr/bin/env python3

import gzip

def convert_mnist_image_to_txt(inf, outf):

    bytestream = gzip.open(inf)
    outfile = open(outf, "w")

    magic = int.from_bytes(bytestream.read(4), byteorder="big") # 2051
    nimages = int.from_bytes(bytestream.read(4), byteorder="big")
    nrows = int.from_bytes(bytestream.read(4), byteorder="big")
    ncols = int.from_bytes(bytestream.read(4), byteorder="big")

    for x in range(nimages):
        bimage = bytestream.read(nrows*ncols)
        image = [str(x) for x in bytearray(bimage)]
        iout = "\t".join(image)
        print(iout, file=outfile)

    bytestream.close()

    return 0


def convert_mnist_label_to_txt(inf, outf):

    bytestream = gzip.open(inf)
    outfile = open(outf, "w")

    magic = int.from_bytes(bytestream.read(4), byteorder="big") # 2051
    nlabels = int.from_bytes(bytestream.read(4), byteorder="big")

    for x in range(nlabels):
        blabel = bytestream.read(1)
        label = int.from_bytes(blabel, byteorder="big")
        print(label, file=outfile)

    bytestream.close()

    return 0


def main():

    import argparse

    parser = argparse.ArgumentParser(description='Convert MNIST to TXT')
    parser.add_argument('-i', '--images', type=str, help="images file", required=True)
    parser.add_argument('-l', '--labels', type=str, help="labels file", required=True)
    parser.add_argument('-o', '--out_prefix', type=str, help="outfiles name prefix", required=True)

    args = parser.parse_args()

    convert_mnist_image_to_txt(args.images, args.out_prefix + "images.txt")
    convert_mnist_label_to_txt(args.labels, args.out_prefix + "labels.txt")

    return 0

if __name__ == '__main__':
    main()
