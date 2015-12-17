import numpy as np
import sys
sys.path.append('../')
import minibatch_generators as mbg


def test_balanced_idxs():
    '''
    Testing for when the classes are balanced - we should just see one idx
    for each index position returned
    '''
    Y = [0, 0, 1, 1, 2, 2]
    ans = list(mbg.balanced_idxs_iterator(Y))
    assert len(Y) == len(ans)
    assert set(ans) == {0, 1, 2, 3, 4, 5}


def test_imbalanced_idxs():
    '''
    Testing for when the class labels are imbalanced
    '''
    Y = [0, 1, 1, 2, 2, 3, 3, 3, 3, 3]
    ans = list(mbg.balanced_idxs_iterator(Y))

    # we should see each idx at least once
    assert set(ans) == set(range(len(Y)))

    # each class should be seen the same number of times
    provided_classes_counts = np.bincount([Y[idx] for idx in ans])
    assert provided_classes_counts.min() == provided_classes_counts.max()

    # the total number of items returned should be equal to the size of the
    # largest class times the number of classes
    assert len(ans) == np.bincount(Y).max() * np.unique(Y).shape[0]


def check_minibatches(list_of_minibatches, minibatch_size):
    '''
    Checks the minibatches are of the correct lengths
    '''
    # all but the last minibatch should be at the specified length
    for ll in list_of_minibatches[:-1]:
        assert len(ll) == minibatch_size

    final_minibatch = list_of_minibatches[-1]
    assert len(final_minibatch) <= minibatch_size


def test_minibatch_idx_iterator():
    '''
    Testing the minibatches of idxs are made correctly
    minibatch_idx_iterator(Y, minibatch_size, randomise, balanced)
    '''

    # simplest case - classes are already balanced and minibatch_size == len(Y)
    Y = [0, 0, 1, 1, 2, 2]
    ans = list(mbg.minibatch_idx_iterator(
        Y, 6, randomise=False, balanced=False))

    # ans is a list of lists, the first of which should contain everything
    assert len(ans) == 1
    assert set(ans[0]) == {0, 1, 2, 3, 4, 5}

    # now with randomisation
    ans = list(mbg.minibatch_idx_iterator(Y, 6, randomise=True, balanced=False))
    assert len(ans) == 1
    assert set(ans[0]) == {0, 1, 2, 3, 4, 5}


def test_imbalanced_minibatch_idx_iterator():
    # now with imbalanced classes
    Y = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    ans = list(mbg.minibatch_idx_iterator(
        Y, 10, randomise=False, balanced=False))
    assert len(ans) == 1
    assert set(ans[0]) == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}


    ans = list(mbg.minibatch_idx_iterator(
        Y, 10, randomise=False, balanced=True))

    # simple minibatch check
    check_minibatches(ans, 10)

    # classes should be approximately balanced
    for an in ans:
        class_counts = np.bincount([Y[xx] for xx in an])
        assert class_counts.max() - class_counts.min() <= 1

    # all idxs should be returned at least once
    all_idxs = [yy for xx in ans for yy in xx]
    assert set(all_idxs) == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}


def test_minibatch_iterator():
    '''
    minibatch_iterator(X, Y, minibatch_size, randomise=False, balanced=False,
            x_preprocesser=lambda x:x, stitching_function=lambda x: np.array(x),
            threading=False, num_cached=128):
    '''

    # check X and Y returned correspond
    X = [-1, -2, -3, -4, -5, -6]
    Y = [0, 1, 2, 3, 4, 5]

    mb_x, mb_y = next(mbg.minibatch_iterator(X, Y, len(X)))
    assert all(mb_x == X)
    assert all(mb_y == Y)

    # check the same if we split up into batches
    all_x = []
    all_y = []

    for tmp_x, tmp_y in mbg.minibatch_iterator(X, Y, 5):
        all_x += list(tmp_x)
        all_y += list(tmp_y)

    assert all_x == X
    assert all_y == Y


def test_minibatch_iterator2():
    X = [-1, -2, -3, -4, -5, -6, -7]
    Y = [0, 1, 1, 2, 2, 2, 5]
    for tmp_x, tmp_y in mbg.minibatch_iterator(X, Y, 5, balanced=True):
        print tmp_x, tmp_y


def test_threading():
    import time

    X = [-1] * 50
    Y = [0] * 50
    delay = 0.05

    def augmenter(x):
        '''
        Augmenter function which actually just delays for half a second.
        Imagine this is doing some kind of computationally expensive
        pre-processing or data augmentation.
        '''
        time.sleep(delay)
        return x

    # doing without threading:
    tic = time.time()
    for tmp_x, tmp_y in mbg.minibatch_iterator(
            X, Y, 1, x_preprocesser=augmenter, threading=False):
        time.sleep(delay)

    no_thread_time = time.time() - tic
    print "Without threading", no_thread_time

    # doing with threading:
    tic = time.time()
    for tmp_x, tmp_y in mbg.minibatch_iterator(
            X, Y, 1, x_preprocesser=augmenter, threading=True):
        time.sleep(delay)

    thread_time = time.time() - tic
    print "With threading", time.time() - tic

    ratio = no_thread_time / thread_time
    print ratio
    print np.abs(ratio - 2.0)
    assert np.abs(ratio - 2.0) < 0.05


#
#
# test_balanced_idxs()
# test_imbalanced_idxs()
# test_minibatch_idx_iterator()
# test_imbalanced_minibatch_idx_iterator()
test_minibatch_iterator()
test_minibatch_iterator2()
test_threading()
