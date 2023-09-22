
import numpy as np
from skimage.transform import resize as imresize
#scipy.misc.imresize

def rescale_np(array):
    a_min, a_max = np.amin(array), np.amax(array)
    return (array - a_min) / (a_max - a_min)


def corr_coeff(map1, map2):  # tested \,
    # map1 and map2 are 2 real-valued matrices numpy
    assert(np.amin(map1) >= 0)
    assert(np.amin(map2) >= 0)
    map1 = map1.flatten()
    map2 = map2.flatten()

    # normalize both maps
    map1 = (map1 - np.mean(map1)) / np.std(map1)
    map2 = (map2 - np.mean(map2)) / np.std(map2)

    return np.corrcoef(map2, map1)[0, 1]


def nss(salmap, fixations, size=None):  # Tested \,
    # normalized scanpath saliency
    # salmap is the saliency map
    # fixmap is the human fixation map (binary matrix)
    assert(np.amin(salmap) >= 0)
    if size is not None:
        salmap = imresize(salmap, size)

    # normalize saliency map
    salmap = (salmap - np.mean(salmap)) / np.std(salmap, ddof=1)

    # mean value at fixation locations
    return np.mean(salmap[fixations.astype(bool)])


def auc_judd(salmap, fixations, size=None, jitter=True):  # Tested \,
    # https://github.com/cvzoya/saliency/blob/master/code_forMetrics/AUC_Judd.m
    # created: Tilke Judd, Oct 2009
    # updated: Zoya Bylinskii, Aug 2014
    #
    # This measures how well the salmap of an image predicts the ground
    # truth human fixations on the image.
    #
    # ROC curve created by sweeping through threshold values
    # determined by range of saliency map values at fixation locations;
    # true positive (tp) rate correspond to the ratio of saliency map values
    # above threshold at fixation locations to the total number of fixation
    # locations false positive (fp) rate correspond to the ratio of saliency map
    # values above threshold at all other locations to the total number of
    # posible other locations (non-fixated image pixels)
    #
    # salmap is the saliency map
    # fixationMap is the human fixation map (binary matrix)
    # jitter = 1 will add tiny non-zero random constant to all map locations
    # to ensure ROC can be calculated robustly (to avoid uniform region)
    # if toPlot=1, displays ROC curve

    fixations = fixations.astype(bool)
    n_fixations = int(np.sum(fixations))
    if n_fixations == 0:
        return 0.

    # make the salmap the size of the image of fixationMap
    if size is not None:
        salmap = imresize(salmap, size)

    # jitter saliency maps that come from saliency models that have a lot of
    # zero values.  If the saliency map is made with a Gaussian then it does
    # not need to be jittered as the values are varied and there is not a large
    # patch of the same value. In fact jittering breaks the ordering
    # in the small values!
    if jitter:
        # jitter the saliency map slightly to distrupt ties of the same numbers
        salmap = salmap + np.random.rand(*salmap.shape) / 10000000

    # normalize saliency map
    salmap = rescale_np(salmap)
    s_th = salmap[fixations]
    s_th = s_th.flatten().tolist()
    s_th.sort(reverse=True)  # sort sal map values, to sweep through values

    n_pixels = salmap.size
    tp = np.zeros(n_fixations + 2)
    fp = np.zeros(n_fixations + 2)
    tp[-1] = 1
    fp[-1] = 1

    for idx, thresh in enumerate(s_th, 1):
        # total number of sal map values above threshold
        aboveth = np.sum(salmap >= thresh)
        # ratio sal map values at fixation locations above threshold
        tp[idx] = idx / n_fixations
        # ratio other sal map values above threshold
        fp[idx] = (aboveth - idx) / (n_pixels - n_fixations)

    return np.trapz(tp, fp)


def kld(map1, map2, eps=4e-16):  # \, tested
    assert(np.amin(map1) >= 0)
    assert(np.amin(map2) >= 0)
    map1 /= np.sum(map1)
    map2 /= np.sum(map2)
    return np.sum(map2 * np.log(eps + map2 / (map1 + eps)))


def sim(map1, map2):
    assert(np.amin(map1) >= 0)
    assert(np.amin(map2) >= 0)

    if np.sum(map1) > 0:
        map1 = rescale_np(map1)
        map1 /= np.sum(map1)
    if np.sum(map1) > 0:
        map2 = rescale_np(map2)
        map2 /= np.sum(map2)

    diff = np.minimum(map1, map2)
    return np.sum(diff)
