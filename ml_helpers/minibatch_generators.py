import numpy as np
import itertools


def balanced_idxs_iterator(Y, randomise=False):
    '''
    Iterates over the index positions in Y, such that at the end
    of a complete iteration the same number of items will have been
    returned from each class.
    Every item in the biggest class(es) is returned exactly once.
    Some items from the smaller classes will be shown more than once.
    '''
    biggest_class_size = np.bincount(Y).max()

    # create a cyclic generator for each class
    generators = {}
    for class_label in np.unique(Y):
        idxs = np.where(Y == class_label)[0]

        if randomise:
            idxs = np.random.permutation(idxs)

        generators[class_label] = itertools.cycle(idxs)

    # number of loops is defined by the largest class size
    for _ in range(biggest_class_size):
        for generator in generators.itervalues():
            data_idx = generator.next()
            yield data_idx


def minibatch_idx_iterator(Y, minibatch_size, randomise, balanced):

    if balanced:
        iterator = balanced_idxs_iterator(Y, randomise)

        # the number of items that will be yielded from the iterator
        num_to_iterate = np.bincount(Y).max() * np.bincount(Y).shape[0]
    else:
        if randomise:
            iterator = iter(np.random.permutation(xrange(len(Y))))
        else:
            iterator = iter(range(len(Y)))

        # the number of items that will be yielded from the iterator
        num_to_iterate = len(Y)

    num_minibatches = int(np.ceil(float(num_to_iterate) / float(minibatch_size)))

    for _ in range(num_minibatches):
        # use a trick to ensure we return a partial minibatch at the end...
        idxs = [next(iterator, None) for _ in range(minibatch_size)]
        yield [idx for idx in idxs if idx is not None]


def threaded_gen(generator, num_cached=128):
    '''
    Threaded generator to multithread the data loading pipeline
    code from daniel, he got it from a chatroom or something...
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
    Could use x_preprocessor for data augmentation for example (making use of partial)
    '''
    iterator = minibatch_idx_iterator(Y, minibatch_size, randomise, balanced)

    if threading:
        # return a version of this generator, wrapped in the threading code
        itr = minibatch_iterator(X, Y, minibatch_size, randomise=randomise,
            balanced=balanced, x_preprocessor=x_preprocessor,
            stitching_function=stitching_function, threading=False)

        for xx in threaded_gen(itr, num_cached):
            yield xx

    # don't really need the else but keeping for readability
    else:

        for minibatch_idxs in iterator:

            # extracting the Xs, and appling preprocessing (e.g augmentation)
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
    temp = np.concatenate(im_list, 3)
    temp = temp.transpose((3, 2, 0, 1))
    return temp.astype(np.float32)
