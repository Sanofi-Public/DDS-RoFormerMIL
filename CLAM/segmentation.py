import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as scilabel
from skimage import morphology as morph
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation
from tiatoolbox.wsicore import wsireader


def simple_tissue_segmentation(
    img_rgb, min_object_size, kernel_size_tophat=3, connectivity_dilation=8
):
    """Simple pipeline that converts an rgb img to gray,
    then applies a top hat filter followed by otsu binarization, dilation and hole filling.

    Args:
        img_rgb (np.array): np array ocntaining the rgb img
        min_object_size (float): objects with area smaller than this value will be removed
        kernel_size_tophat (int, optional): filter size of tophat. Defaults to 3.
        connectivity_dilation (int, optional): connectivity value used in dilation. Defaults to 8.

    Returns:
        np.array: binary image with the computed tissue segmentation
    """
    img_gray = rgb2gray(img_rgb)

    filterSize = (kernel_size_tophat, kernel_size_tophat)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)

    tophat_img = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)

    img_bin = binarize(
        1 - tophat_img, connectivity=connectivity_dilation, postprocess=True
    )
    img_bin, labels = remove_small(img_bin, min_object_size)

    return img_bin


def binarize(gray, connectivity=8, postprocess=False):
    """
    For a given gray image it thresholds it with otsu.
    If postprocess is True it will dilate the binary image with a selem of size connectivity.
    """
    thresh = threshold_otsu(gray)
    # thresh=threshold/255
    # print('thresh', thresh)
    BW = gray < thresh
    if postprocess:
        BW = binary_dilation(BW, footprint=np.ones((connectivity, connectivity)))
        BW = binary_fill_holes(BW)

    return BW


def remove_small(BW, min_object_size):
    """
    removes all connected components with a size less than min_object_size

    """
    labels = scilabel(BW)[0]
    BW = morph.remove_small_objects(labels, min_size=min_object_size)
    labels = scilabel(BW)[0]
    return (BW != 0), labels


def simple_segmentation(pimg, mag_thumb, min_area_ratio=0.0003):
    """
    mag_thumb is the pyramid level, corresponding to the seg_level variable in the CLAM code
    """

    imgptr = wsireader.OpenSlideWSIReader(pimg)
    factor = imgptr.info.objective_power / mag_thumb  # from thumb to HR
    he_lr = imgptr.slide_thumbnail(resolution=mag_thumb, units="level")
    min_object_size = min_area_ratio * np.prod(he_lr.shape[:2])
    mask_tissue_lr = simple_tissue_segmentation(he_lr, min_object_size=min_object_size)

    return mask_tissue_lr
