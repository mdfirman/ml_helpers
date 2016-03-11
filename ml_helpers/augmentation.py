'''
A collection of functions to perform augmentations of images, e.g. for
deep learning algorithms.

Some of these (e.g. randn_crop, random_rotation) assume that we have a
rectangular region of interest specified on the image.

See examples/augment.py for some usage examples.
'''
import numpy as np
from skimage.transform import rotate

def randn_crop(im, top, bottom, left, right, sd=10):
    '''
    Returns some random crops of an image, with offsets from the 'true' crop
    sampled from a normal distribution.
    This is good for if you already have a bounding box of the object in
    question, and you want to return perturbed versions of this.

    Parameters:
    ------------------
    im:
        rgb or greyscale image
    sd:
        standard deviation of the crop offsets in pixels
    top, bottom, left, right:
        scalars giving the positions in pixels of the region of interest in the
        image. If specified, the randomly generated crops are done relative to
        these locations
    '''
    crop_top = max(0, top + np.random.randn() * sd)
    crop_left = max(0, left + np.random.randn() * sd)

    height, width = im.shape[0], im.shape[1]
    crop_bottom = min(height, bottom + np.random.randn() * sd)
    crop_right = min(width, right + np.random.randn() * sd)

    return im[crop_top:crop_bottom, crop_left:crop_right]


def random_crop(im, end_width=224, end_height=224):
    '''
    Using alexnet cropping scheme, which takes crops from within the image

    Parameters:
    ------------------
    im:
        rgb or greyscale image
    end_width, end_height:
        the desired width and height of the final returned crop
    '''
    top = np.random.randint(0, im.shape[0] - end_height)
    left = np.random.randint(0, im.shape[1] - end_width)
    return im[top:top+end_height, left:left+end_width]


def random_flip(im, leftright=True, topbottom=True):
    '''
    Flips image horizontally and/or vertically with probability 0.5

    Parameters:
    ------------------
    im:
        rgb or greyscale image
    leftright, topbottom:
        booleans dictating what the permissable flips are
    '''
    lr = int(leftright and np.random.rand() < 0.5) * -2 + 1
    tb = int(topbottom and np.random.rand() < 0.5) * -2 + 1
    return im[::lr, ::tb]


def random_colour_transform(im, rgb_eigval, rgb_eigvec, sd=0.1,
        clip=(0, 1.0)):
    '''
    Performs a colourspace warping, according to maybe alexnet (or baidu)
    Does not do any clipping, so returned image may fall out

    Parameters:
    ------------------
    im:
        rgb or greysacle image
    rgb_eigval, rgb_eigvec:
        the eigenvectors and eigenvalues which we will use to perform the
        transformations
    sd:
        the standard deviations of the rgb transforms applied (see [1])
    clip:
        if provided, clips the final image to lie within the stated values

    [1] ImageNet Classification with Deep Convolutional Neural Networks
    '''
    alpha = np.random.randn(3) * sd
    scale_factor = alpha * np.real(rgb_eigval)
    scaled_im = im + rgb_eigvec.dot(scale_factor)[None, None, :]

    if clip is None:
        return scaled_im
    else:
        return np.clip(scaled_im, *clip)


def random_rotation(im, top, bottom, left, right, all_rotations=False,
        sd=10):
    '''
    Returns a randomly rotated version of im about the centre of the specified
    crop.

    Parameters:
    ------------------
    im:
        rgb or greysacle image
    top, bottom, left, right:
        These specify the region of interest in the image, and are used to
        rotate the image about the centre of the region of interest
    all_rotations:
        If true, rotation angle is uniformly sampled in [0, 360]; otherwise,
        the angle is sampled from a normal distribution.
    sd:
        Standard deviation of the rotation in degrees; only used if
        all_rotations is False.

    '''
    if all_rotations:
        angle = np.random.rand() * 360
    else:
        angle = np.random.randn() * sd

    centre = ((left+right)/2, (top + bottom)/2)
    return rotate(im, angle=angle, resize=False, center=centre, order=1)
