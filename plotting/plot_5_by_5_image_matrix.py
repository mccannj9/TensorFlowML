#! /usr/bin/env python3

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


def main():
    pass


if __name__ == '__main__':
    main()
