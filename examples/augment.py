
from skimage.io import imread, imsave
import yaml
import matplotlib.pyplot as plt
from scipy.misc import imresize
from scipy.linalg import eig
from ml_helpers import augmentation as aug
import numpy as np
import os
from time import time

def plot_grid(ims, savepath=None):
    '''
    Displays and optionally saves a list of 25 images in a 5x5 grid.
    Uses matplotlib which makes it quite slow, but this allows for different
    size images, and also for titles and labels to be added easily if needed.
    '''
    # ims = [imresize(im, (256, 256)) for im in ims]

    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for idx, im in enumerate(ims[:25]):
        plt.subplot(5, 5, idx+1)
        plt.imshow(im)
        plt.axis('off')

    if savepath:
            plt.savefig(savepath)

data_path = 'data/9c02dca8ac0062ada81b07bb70e6dbff'

savepath = './output_augmentations/'
if not os.path.exists(savepath):
    os.makedirs(savepath)

im = imread(data_path + '.jpg').astype(np.float32) / 255.0
crop = yaml.load(open(data_path + '_crop.yaml'))['crop']

newcrop = {'left': crop['x'],
           'right': crop['x'] + crop['width'],
           'top': crop['y'],
           'bottom': crop['y'] + crop['height']}

cropped_im = \
   im[newcrop['top']:newcrop['bottom'], newcrop['left']:newcrop['right']]

############################################
# ROTATIONS
tic = time()
rots = [aug.random_rotation(im, sd=10, **newcrop) for _ in range(24)]
rots = [aug.randn_crop(im, sd=0, **newcrop) for im in rots]
ims = [cropped_im] + rots
print "Rotating took %fs per image" % ((time()-tic)/24.0)
plot_grid(ims, savepath + 'rotations.png')

############################################
# COLOUR TRANSFORMS
rgb_vals = cropped_im.transpose((2, 0, 1)).reshape((3, -1))
rgb_eigval, rgb_eigvec = eig(np.cov(rgb_vals - rgb_vals.mean(1)[:, None]))
rgb_eigvec, rgb_eigval, v = np.linalg.svd(
    rgb_vals - rgb_vals.mean(1)[:, None], full_matrices=False)
rgb_eigval = np.real(rgb_eigval)

tic = time()
ims = [cropped_im] + [aug.random_colour_transform(
    cropped_im, rgb_eigval, rgb_eigvec) for _ in range(24)]
print "Colour transform took %fs per image" % ((time()-tic)/24.0)
plot_grid(ims, savepath + 'cols.png')

############################################
# NORMS
tic = time()
ims = [cropped_im for _ in range(25)]
print "No augmentation took %fs per image" % ((time()-tic)/24.0)
plot_grid(ims, savepath + 'norms.png')

############################################
# CROPS
tic = time()
ims = [cropped_im] + [aug.randn_crop(im, sd=10, **newcrop) for _ in range(24)]
print "Randn cropping took %fs per image" % ((time()-tic)/24.0)
plot_grid(ims, savepath + 'crops.png')

############################################
# CROPS
tic = time()
resized_im = imresize(cropped_im, (256, 256))
ims = [cropped_im] + [aug.random_crop(resized_im) for _ in range(24)]
print "Alexnet cropping took %fs per image" % ((time()-tic)/24.0)
plot_grid(ims, savepath + 'alexnet_crops.png')

############################################
# FLIPS
tic = time()
ims = [cropped_im] + [aug.random_flip(cropped_im) for _ in range(24)]
print "Flipping took %fs per image" % ((time()-tic)/24.0)
plot_grid(ims, savepath + 'flips.png')

############################################
# ALL
tic = time()
def full_trans(im):
    rotated = aug.random_rotation(im, sd=10, **newcrop)
    cropped = aug.randn_crop(rotated, sd=10, **newcrop)
    flipped = aug.random_flip(cropped)
    coloured = aug.random_colour_transform(flipped, rgb_eigval, rgb_eigvec)
    return imresize(coloured, (256, 256))

ims = [cropped_im] + [full_trans(im) for _ in range(24)]
print "Full transform took %fs per image" % ((time()-tic)/24.0)
plot_grid(ims, savepath + 'full.png')
