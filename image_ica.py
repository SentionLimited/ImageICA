# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import sys


def uint8_to_normed_floats(I):
    """
    Convert a uint8 array, I, into a normed array of floats in the range {0,1}.

    Parameters
    ----------
    I : numpy array (uint8)
        Arbitrarily shaped numpy array of data type uint8.

    Returns
    -------
    numpy array (floats)
        Normalised numpy array of floats.

    """
    if I.dtype == np.uint8:
        return I.astype(float)/255
    else:
        raise Exception('Array must be of dtype uint8.')
        
def preprocess_img(I):
    """
    Convert uint8 image, I, into an array of normalised floats in the range 
    {0,1}, and enhance lumininace by scaling all channels by the normalised 
    greyscale Lab luminosity.

    Parameters
    ----------
    I : numpy array (uint8)
        Arbitrarily shaped numpy array of data type uint8.

    Returns
    -------
    I : numpy array (floats)
        Normalised and enhanced image.

    """
    
    # Extract and normalise luminosity image from Lab colour format
    Lab = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
    L = uint8_to_normed_floats(Lab[:,:,0])
    
    # Normalise RGB image and scale all channels by the greyscale luminosity
    I = uint8_to_normed_floats(I)
    L = np.repeat(L[:,:,np.newaxis], 3, axis=2)
    I = L * I
    
    return I

def extract_img_patches(I, PATCH_DIM, NUM_PATCHES):
    """
    Extract an n = NUM_PATCHES number of randomly chosen image patches, where  
    each patch has shape, PATCH_DIM, and is extracted from the image I.

    Parameters
    ----------
    I : numpy array (floats)
        Image from which the random patches are extracted.
    PATCH_DIM : numpy array (ints)
        Array of shape {H x W x C} that defines the height, H, and width, W, 
        of the image patches (C is the number of colour channels, but is of no
        importance in this function).
    NUM_PATCHES : int
        Number of random patches to be extracted.

    Returns
    -------
    img_patches : numpy array {floats}
        The {p  x NUM_PATCHES} array of all vectorised image patches, and 
        where p is the total number of pixel values in a patch array.
    """
    # Initialise output array of randomly sampled image patches
    img_patches = np.zeros([np.prod(PATCH_DIM), NUM_PATCHES])
    
    # Calculate the {x,y} border limits for sampling a patch from the image
    xlim = I.shape[1] - PATCH_DIM[1]
    ylim = I.shape[0] - PATCH_DIM[0]
    
    # Generate and store the random set of image patches
    for i in range(NUM_PATCHES):
        
        # Create random [x, y] image index that defines the top-left pixel of 
        # current random image patch
        xind = int(np.floor(np.random.rand(1) * xlim))
        yind = int(np.floor(np.random.rand(1) * ylim))
        
        # Calculate the range of x and y indices that sample current patch
        xrange = np.arange(xind, xind + PATCH_DIM[1])
        yrange = np.arange(yind, yind + PATCH_DIM[0])
        
        # Store image patch in vectorised form
        img_patches[:,i] = I[yrange[:,np.newaxis], xrange, :].flatten()
        
    return img_patches

def mean_centred_img_patches(img_patches):
    """
    Given the input array, img_patches, of dimension {p x NUM_PATCHES}, 
    centre all p-element column vectors onto the mean vector image patch.

    Parameters
    ----------
    img_patches : numpy array (floats)
        The {p x NUM_PATCHES} array of randomly samples p-element image patch 
        vectors.

    Returns
    -------
    centred_img_patches : numpy array (floats)
        The mean-centred {p x NUM_PATCHES} array of p-elements image patches.

    """
    # Mean image patch in vectorised form
    mean_pixel_vals = np.mean(img_patches, axis=1)
    
    # Centre all image patches onto the mean image patch
    centred_img_patches = img_patches - mean_pixel_vals[:, np.newaxis]
    
    return centred_img_patches
    
def weight_init(PATCH_DIM, NUM_MODES):
    """
    Randomly initialise the vector of weights to be used in the numerical
    search procedure when finding the image ICA decomposition. 

    Parameters
    ----------
    PATCH_DIM : numpy array (int)
        Array of shape {H x W x C} that defines the height, H, width, W, and
        number, C, of colour channels for an image patch.
    NUM_MODES : int
        Number of independent modes into which the image will be decomposed.

    Returns
    -------
    nump array {floats}
        The randomly initialised column vector of weights.
        
    """
    # Random weights will be in range {0, UNIT_SCALING}
    UNIT_SCALING = 1e-2; 
    
    # Total number of pixels in image patch
    num_pixels_in_patch = np.prod(PATCH_DIM) 
    
    # Allocate 2 random weights per pixel per mode 
    w = UNIT_SCALING * np.random.rand(2 * NUM_MODES * num_pixels_in_patch)
    
    return w[:, np.newaxis]

def generate_inverse_mapping(w, centred_img_patches, NUM_MODES):
    """
    Extract the inverse mapping from the image space to the space of 
    independent sources.

    Parameters
    ----------
    w : numpy array (floats)
        Column vector of model weights, used to construct mapping.
    centred_img_patches : numpy array (floats)
        The {p x NUM_PATCHES} array of all centred vectorised image patches.
    NUM_MODES : int
        Number of independent modes into which the image will be decomposed.

    Returns
    -------
    inverse_mapping : numpy array (floats)
        The {NUM_MODES x p} matrix transform that maps the image patches to
        the desired sources.

    """
    # Shape of the matrix that maps the array of sources onto the space of
    # centred image patches
    mapping_shape = [centred_img_patches.shape[0], NUM_MODES]
    
    # Central index of the weight vector
    mid_ind = int(w.size/2)
    
    # The matrix transformation from sources is complex, such that half of the 
    # weights in w are scattering amplitudes, and half are scattering phases.
    amp = np.reshape(w[:mid_ind], mapping_shape)
    phase = np.reshape(w[mid_ind:], mapping_shape)
    
    # Transformation that maps sources to centred image patches
    mapping = amp * np.exp(-1j * phase)
    
    # Inverse transform that maps centred image patches to sources
    inverse_mapping = np.linalg.pinv(mapping)
    
    return inverse_mapping

def map_patches_to_sources(inverse_mapping, centred_img_patches):
    """
    Given the inverse mapping from image to source space, calculate the array
    of modal sources.

    Parameters
    ----------
    inverse_mapping : numpy array (floats)
        The {NUM_MODES x p} matrix transform that maps the image patches to
        the desired sources.
    centred_img_patches : numpy array (floats)
        The {p x NUM_PATCHES} array of all centred vectorised image patches.

    Returns
    -------
    numpy array (floats)
        The {NUM_MODES x NUM_PATCHES} array of sources.

    """
    return inverse_mapping @ centred_img_patches

def cov(sources):
    """
    Given the array of sources for all image patches, calculate the covariance
    array between all modes.

    Parameters
    ----------
    sources : numpy array (floats)
        The {NUM_MODES x NUM_PATCHES} array of sources.

    Returns
    -------
    numpy array (floats)
        The {NUM_MODES x NUM_MODES} covariance array between all modes.

    """
    return (sources @ sources.T)/sources.shape[1]

def get_KL(v1, v2):
    """
    Given two arbitrary vectors, v1 and v2, for which all elements are >= 0, 
    and for which both are of equal shape, calculate the KL divergence if each
    vector were to represent an unnnormalised discrete probability 
    distribution. [It is permissable for v1 and v2 to be non-vector arrays of
    equal shape, but they will be flattened (in a row-major order) and treated
    as vectors].

    Parameters
    ----------
    v1 : numpy array {floats}
        Vector of values >= 0.
    v2 : numpy array {floats}
        Vector of values >= 0.

    Returns
    -------
    KL : floats
        The KL-divergence of the two unnormalised probability distribubtions.

    """

    assert np.all(np.array([type(v1), type(v2)]) == 2*[np.ndarray]), (
        'Both inputs must be numpy arrays')
    assert v1.shape == v2.shape, (
        'Both inputs must have equivalent shape.')
    assert np.min((v1,v2)) >= 0, (
        'Input vectors must have all elements >= 0')
    
    # Prevent NaNs by replacing null values with machine float resolution
    zero = np.finfo(float).resolution
    v1[np.where(np.abs(v1) == 0)] = zero
    v2[np.where(np.abs(v2) == 0)] = zero
    
    # Calculate the implied probability distributions
    p1 = v1/np.sum(v1)
    p2 = v2/np.sum(v2)
    
    # KL-divergence
    KL = np.sum(p1*np.log(p1) - p1*np.log(p2))
    
    return KL

def pairwise_modal_divergences(sources):
    """
    Use a symmetrised function of the KL-divergence as a multi-dimensional 
    distance measure between all pairwise combinations of the absolute 
    magnitudes of source modes.

    Parameters
    ----------
    sources : numpy array (floats)
        The {NUM_MODES x NUM_PATCHES} array of sources.

    Returns
    -------
    divergences : numpy array {floats}
        Vector of KL-divergences between all pairs of source-mode magnitudes.

    """
    # Define an index iterator
    def iterate_ind():
        ind = 0
        while True:
            yield ind
            ind +=1
    
    # Absolute magnitude of source array
    abs_sources = np.abs(sources)
    
    num_modes = sources.shape[0] 
    num_mode_pairs = math.comb(num_modes, 2) 
    divergences = np.zeros(num_mode_pairs)
    ind = iterate_ind()
    for i in range(num_modes):
        for j in range(i+1, num_modes):
            
            # Both KL-divergences for a given set of vectors
            KL1 = get_KL(abs_sources[i,:], abs_sources[j,:])
            KL2 = get_KL(abs_sources[j,:], abs_sources[i,:])
            
            # Store symmetrised KL-divergence
            pair_index = next(ind)
            divergences[pair_index] = KL1 + KL2
    
    return divergences

def err(sources, source_cov):
    """
    Extract the error vector, F, of all errors to be minimised in the 
    numerical search procedure.

    Parameters
    ----------
    sources : numpy array (floats)
        The {NUM_MODES x NUM_PATCHES} array of sources.
    source_cov : numpy array (floats)
        The {NUM_MODES x NUM_MODES} covariance array between all modes.

    Returns
    -------
    F : numpy array (floats)
        Column vector of all errors.

    """
    
    # The target source covariance array is the identity
    I = np.eye(source_cov.shape[0])
    cov_err_real = (source_cov.real - I).flatten()
    cov_err_imag = source_cov.imag.flatten()
    
    # The target symmetric divergence between all modes is 2
    TARGET_DIVERGENCE = 2.
    divergences = pairwise_modal_divergences(sources)
    divergence_err = divergences - TARGET_DIVERGENCE
    
    F = np.vstack((cov_err_real[:, np.newaxis],\
                   cov_err_imag[:, np.newaxis],\
                 divergence_err[:, np.newaxis]))
    return F

def jac(w, centred_img_patches, F, NUM_MODES):
    """
    The Jacobian of the numerical search procedure.

    Parameters
    ----------
    w : numpy array (floats)
        Column vector of model weights, used to construct mapping.
    centred_img_patches : numpy array (floats)
        The mean-centred {p x NUM_PATCHES} array of p-elements image patches.
    F : numpy array (floats)
        Column vector of all errors.
    NUM_MODES : int
        Number of independent modes into which the image will be decomposed.

    Returns
    -------
    J : numpy array (floats)
        The Jacobian for the current error vector and set of weights.

    """
    
    # Initialise numerical perturbation and Jacobian array
    PERT = 1e-15
    num_var = w.size
    num_err = F.size
    J = np.zeros([num_err, num_var])
    
    # Iterate over all weights and populate Jacobian
    for i in range(num_var):
        
        w_pert = w.copy()
        w_pert[i] = w[i] + PERT
        inverse_mapping_pert = generate_inverse_mapping(w_pert, centred_img_patches, NUM_MODES)
        sources_pert = map_patches_to_sources(inverse_mapping_pert, centred_img_patches)
        source_cov_pert = cov(sources_pert)
        
        dF = err(sources_pert, source_cov_pert) - F
        J[:,[i]] = dF/PERT
    
    return J

def progress(err_vec, iteration, total_iterations):
    """
    Print the progress of the iterative numerical search procedure.

    Parameters
    ----------
    err_vec : numpy array (float)
        Column vector of all errors.
    iteration : int
        Current iteration.
    total_iterations : int
        Total number of iterations.

    """    
    normed_err = np.linalg.norm(err_vec, 2)    
    perc_progress =  100*(iteration+1)/total_iterations;                        
    
    # Format and print progress to the console
    sys.stdout.write('\r     ERROR: %.2f | PROGRESS: %d/%d [%d%%] ' 
                     % (normed_err, iteration+1, total_iterations, perc_progress)
                     )
    sys.stdout.flush() 

def save_model(SAVE_DIR, inverse_mapping, PATCH_DIM, NUM_MODES):
    """
    Save model to a given directory, SAVE_DIR. If save path does not exist, 
    the directory will be created.

    Parameters
    ----------
    SAVE_DIR : text
        Path of save directory.
    inverse_mapping :numpy array (floats)
        The {NUM_MODES x p} matrix transform that maps the image patches to
        the desired sources.
    PATCH_DIM : numpy array (int)
        Array of shape {H x W x C} that defines the height, H, width, W, and
        number, C, of colour channels for an image patch.
    NUM_MODES : int
        Number of independent modes into which the image will be decomposed.

    """
    
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    
    np.savetxt(SAVE_DIR+'real.csv', inverse_mapping.real, delimiter=',', fmt='%1.21f')
    np.savetxt(SAVE_DIR+'imag.csv', inverse_mapping.imag, delimiter=',', fmt='%1.21f')
    np.savetxt(SAVE_DIR+'patch_dim.csv', PATCH_DIM, delimiter=',', fmt='%i')
    np.savetxt(SAVE_DIR+'num_modes.csv', np.array([NUM_MODES]), fmt='%i')
    
def load_model(LOAD_DIR):
    """
    Load model from a given directory, LOAD_DIR.

    Parameters
    ----------
    LOAD_DIR : text
        Path of load directory.

    Returns
    -------
    inverse_mapping :numpy array (floats)
        The {NUM_MODES x p} matrix transform that maps the image patches to
        the desired sources.
    PATCH_DIM : numpy array (int)
        Array of shape {H x W x C} that defines the height, H, width, W, and
        number, C, of colour channels for an image patch.
    NUM_MODES : int
        Number of independent modes into which the image will be decomposed.

    """
    
    inverse_mapping_real = np.loadtxt(LOAD_DIR+'real.csv', delimiter=',')
    inverse_mapping_imag = np.loadtxt(LOAD_DIR+'imag.csv', delimiter=',')
    inverse_mapping = inverse_mapping_real + 1j*inverse_mapping_imag
    
    PATCH_DIM = list(np.loadtxt(LOAD_DIR+'patch_dim.csv').astype(int))
    NUM_MODES = int(np.loadtxt(LOAD_DIR+'num_modes.csv'))
    
    return inverse_mapping, PATCH_DIM, NUM_MODES
    
def extract_all_patches_from_img(I, PATCH_DIM):
    """
    Beginning from the top-left corner of the image, I, extract all image 
    patches of shape, PATCH_DIM, that can be yielded by scanning top-to-bottom
    and left-to-right. The image is not padded, and thus the total sample of
    patches will generate a truncated version of I along the bottom-most and
    right-most border. The number of truncated pixels is determined by the 
    patch size given by PATCH_DIM.

    Parameters
    ----------
    I : numpy array (floats)
        Normalise image with values in range {0,1}.
    PATCH_DIM : numpy array (int)
        Array of shape {H x W x C} that defines the height, H, width, W, and
        number, C, of colour channels for an image patch.

    Returns
    -------
    all_patches : numpy array (floats)
        The array of dimensions {p x n}, where p is the number of pixel values
        in a patch, and n is the total number of patches that can be extracted
        from I. This array thus stores all patches by column in vector form.
        
    num_patches_in_x : int
        Number of patches that can fit along the width of I.
        
    num_patches_in_y : int
        Number of patches that can fit along the height of I.

    """
    # The right- and bottom-most limit for the top-left corner of patch
    xrange = np.arange(I.shape[1] - PATCH_DIM[1])
    yrange = np.arange(I.shape[0] - PATCH_DIM[0])
    
    # The number of patches that can be fit into the width and height of I
    num_patches_in_x = xrange.size
    num_patches_in_y = yrange.size
    
    # The total number of patches that can be extracted from I
    total_num_patches = num_patches_in_x * num_patches_in_y 
    
    # Initialise array of all vectorised patches
    all_patches = np.zeros([np.prod(PATCH_DIM), total_num_patches])
    
    # Define an index iterator
    def iterate_ind():
        ind = 0
        while True:
            yield ind
            ind +=1
    
    # Iterate over all permissable locations in I and extract patches in 
    # row-major order, to comply with the numpy convention
    i = iterate_ind()
    for y in range(num_patches_in_y):
        for x in range(num_patches_in_x):
            
            patch_xinds = np.arange(xrange[x], xrange[x] + PATCH_DIM[1])
            patch_yinds = np.arange(yrange[y], yrange[y] + PATCH_DIM[0])
            
            patch_ind = next(i)
            all_patches[:, patch_ind] = I[patch_yinds[:,np.newaxis], patch_xinds, :].flatten()

    return all_patches, num_patches_in_x, num_patches_in_y

def im_shift_norm(I):
    """
    Given an image, I, scale all pixels such that the {min, max} of the entire 
    image is remapped to the range {0,1}.

    Parameters
    ----------
    I : numpy array (floats)
        Image array with arbitrary min and max pixel value.

    Returns
    -------
    numpy array (floats)
        Image with remapped values such that min(I) = 0 and max(I) = 1.

    """
    if not I.dtype == float:
        I.astype(float)
        
    return (I - np.min(I)) / (np.max(I) - np.min(I))

def visualise_solution(I, inverse_mapping, PATCH_DIM, NUM_MODES):
    """
    Given the model variables, output the stack of all modal images derived 
    from the absolute magnitude of the image source array, as well as the final
    appropriately rescaled sum over all modal images. 

    Parameters
    ----------
    I : numpy array (floats)
        Normalise image into range {0,1}.
    inverse_mapping : numpy array (floats)
        The {NUM_MODES x p} matrix transform that maps the image patches to
        the desired sources.
    PATCH_DIM : numpy array (int)
        Array of shape {H x W x C} that defines the height, H, width, W, and
        number, C, of colour channels for an image patch.
    NUM_MODES : int
        Number of independent modes into which the image will be decomposed.

    Returns
    -------
    final_img : numpy array (floats)
        The rescaled sum of all modal images.
    modal_images : numpy array (floats)
        The stack of all modal images, derived from the absolute values of the
        source modes.

    """
    # Extract all patches from the image
    all_img_patches, num_patches_in_x, num_patches_in_y = \
        extract_all_patches_from_img(I, PATCH_DIM)
    
    # Centre all patches onto the mean image patch
    all_img_patches_centred = mean_centred_img_patches(all_img_patches)
    
    # Map all image patches onto source modes
    all_sources = map_patches_to_sources(inverse_mapping, all_img_patches_centred)
    
    # Initialise the stack of all source mode images
    modal_images = np.zeros([num_patches_in_y, num_patches_in_x, NUM_MODES])
    
    # Extract the images generated by the absolute magnitudes of the source modes
    for i in range(NUM_MODES):
        modal_images[:,:,i] = np.reshape(np.abs(all_sources[i,:]),
                                         [num_patches_in_y, num_patches_in_x])
        
    # Generate the summed image over all modes
    summed_modes = np.sum(modal_images, axis=2)
    
    # Shift and normalise the values of the final summed greyscale image
    final_img = im_shift_norm(summed_modes)
    
    return final_img, modal_images 

def save_visualisation(SAVE_DIR, img):
    """
    Save visualisation to a given directory, SAVE_DIR. If save path does not
    exist, the directory will be created.

    Parameters
    ----------
    SAVE_DIR : text
        Path of save directory.
    img : numpy array (uint8)
        Resulting image after running ICA method.

    """
    
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    
    # cv2.imwrite(SAVE_DIR+'vis.png', img)
    plt.imsave(SAVE_DIR+'vis.png', img)
