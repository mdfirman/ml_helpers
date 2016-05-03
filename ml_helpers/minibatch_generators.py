import numpy as np
import itertools


def force_immutable(item):
    '''
    Forces mutable items to be immutable
    '''
    if hasattr(item, '__next__'):
        return tuple(item)
    else:
        return item


def largest_class_size(y):
    '''
    Like np.bincount(xx).max(), but works for structured arrays too
    '''
    return max([np.sum(y == yy) for yy in np.unique(y)])


def balanced_idxs_iterator(Y, randomise=False):
    '''
    Iterates over the index positions in Y, such that at the end
    of a complete iteration the same number of items will have been
    returned from each class.
    Every item in the biggest class(es) is returned exactly once.
    Some items from the smaller classes will be shown more than once.

    Parameters:
    ------------------
    Y:
        numpy array of discrete class labels
    '''
    biggest_class_size = largest_class_size(Y)

    # create a cyclic generator for each class
    generators = {}
    for class_label in np.unique(Y):
        idxs = np.where(Y == class_label)[0]

        if randomise:
            idxs = np.random.permutation(idxs)

        generators[force_immutable(class_label)] = itertools.cycle(idxs)

    # number of loops is defined by the largest class size
    for _ in range(biggest_class_size):
        for generator in generators.itervalues():
            data_idx = generator.next()
            yield data_idx


def minibatch_idx_iterator(Y, minibatch_size, randomise, balanced):
    '''
    Yields arrays of minibatches, defined by idx positions of the items making
    up each minibatch.

    Parameters:
    ------------------
    Y:
        Numpy array of discrete class labels. This is used to balance the
        classes (if required), and to determine the total number of items in
        the full dataset.
    minibatch_size:
        The maximum number of items required in each minibatch
    randomise:
        If true, there are two types of randomisation: Which items are in each
        minibatch, and the order of items in each minibatch
    balanced:
        If true, each minibatch will contain roughly equal items from each
        class
    '''
    if balanced:
        iterator = balanced_idxs_iterator(Y, randomise)

        # the number of items that will be yielded from the iterator
        num_to_iterate = largest_class_size(Y) * np.unique(Y).shape[0]
    else:
        if randomise:
            iterator = iter(np.random.permutation(xrange(len(Y))))
        else:
            iterator = iter(range(len(Y)))

        # the number of items that will be yielded from the iterator
        num_to_iterate = len(Y)

    num_minibatches = int(
        np.ceil(float(num_to_iterate) / float(minibatch_size)))

    for _ in range(num_minibatches):
        # use a trick to ensure we return a partial minibatch at the end...
        idxs = [next(iterator, None) for _ in range(minibatch_size)]
        yield [idx for idx in idxs if idx is not None]


def threaded_gen(generator, num_cached=1000):
    '''
    Threaded generator to multithread the data loading pipeline
    I'm not sure how effective this is for my current setup. I feel I may be
    better off using multiprocessing or something similar...

    Code from:
    https://github.com/Lasagne/Lasagne/issues/12#issuecomment-59494251

    Parameters
    ---------------
    generator:
        generator which probably will yield pairs of training data
    num_cached:
        How many items to hold in the queue. More items means a higher load
        on the RAM.
    '''
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()


def minibatch_iterator(X, Y, minibatch_size, randomise=False, balanced=False,
        x_preprocesser=lambda x:x, stitching_function=lambda x: np.array(x),
        threading=False, num_cached=128):

    '''
    Could use x_preprocessor for data augmentation for example (making use of
    partial)
    '''
    assert len(X) == len(Y)

    if threading:
        # return a version of this generator, wrapped in the threading code
        itr = minibatch_iterator(X, Y, minibatch_size, randomise=randomise,
            balanced=balanced, x_preprocesser=x_preprocesser,
            stitching_function=stitching_function, threading=False)

        for xx in threaded_gen(itr, num_cached):
            yield xx

    else:

        iterator = minibatch_idx_iterator(
            Y, minibatch_size, randomise, balanced)

        for minibatch_idxs in iterator:

            # extracting the Xs, and apply preprocessing (e.g augmentation)
            Xs = [x_preprocesser(X[idx]) for idx in minibatch_idxs]

            # stitching Xs together and returning along with the Ys
            yield stitching_function(Xs), np.array(Y)[minibatch_idxs]


def atleast_nd(arr, n, copy=True):
    '''http://stackoverflow.com/a/15942639/279858'''
    if copy:
        arr = arr.copy()

    arr.shape += (1,) * (4 - arr.ndim)
    return arr


def form_correct_shape_array(X):
    """
    Given a list of images each of the same size, returns an array of the shape
    and data type (float32) required by Lasagne/Theano
    """
    im_list = [atleast_nd(xx, 4) for xx in X]
    try:
        temp = np.concatenate(im_list, 3)
    except ValueError:
        print "Could not concatenate arrays correctly"
        for im in im_list:
            print im.shape,
        raise

    temp = temp.transpose((3, 2, 0, 1))
    return temp.astype(np.float32)
