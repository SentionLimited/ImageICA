#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np

import image_ica as ica

####################### LOAD AND PREPROCESS IMAGE #############################

# Load uint8 image
I = cv2.imread('test.png')
assert I.ndim == 3, ('Image must have 3 channels, even if greyscale')

# Rearrange BGR colour channels into RGB order
if I.shape[2] == 3:
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

# Normalise image into range {0,1} and preprocess to enhance luminance
I = ica.preprocess_img(I)

########################## SCRIPT PARAMETERS ##################################

LEARN_MODEL = 1         # Learn new model for the current image?
SAVE_TO_FILE = 1        # Save learned model to file?
SAVE_DIR = 'Results/'   # Save directory
LOAD_FROM_FILE = 0      # Load previous model from file?
LOAD_DIR = 'Results/'   # Load directory
VISUALISE = 1           # Visualise the solution?

########################## MODEL PARAMETERS ###################################

if LEARN_MODEL == 1:

    NUM_PATCHES = 1000 # Number of random patches to extract from image
    PATCH_DIM = [3,3,I.shape[2]] # Shape of extracted image patches
    NUM_MODES = 6 # Number of independent source modes
    NUM_ITERATIONS = 100 # Number of iterations in numerical search

###################### NUMERICAL SEARCH PROCEDURE #############################
    
    # Extract array of vectorised image patches
    img_patches = ica.extract_img_patches(I, PATCH_DIM, NUM_PATCHES)
    
    # Centre all image patches onto the mean image patch
    centred_img_patches = ica.mean_centred_img_patches(img_patches)
    
    # Initialise weights, inverse mapping, sources, covariances and error vector
    w = ica.weight_init(PATCH_DIM, NUM_MODES)
    inverse_mapping = ica.generate_inverse_mapping(w, centred_img_patches, NUM_MODES)
    sources = ica.map_patches_to_sources(inverse_mapping, centred_img_patches)
    source_cov = ica.cov(sources)
    F = ica.err(sources, source_cov)
    
    print('\n * LEARNING MODEL     ')
    print(' ------------------------------------------')
    print(' IN. ERROR:', '%.2f'%np.linalg.norm(F,2))
    
    # Iterate model through numerical search procedure
    for i in range(NUM_ITERATIONS):
        
        J = ica.jac(w, centred_img_patches, F, NUM_MODES)
        w = w - np.linalg.pinv(J) @ F
        inverse_mapping = ica.generate_inverse_mapping(w, centred_img_patches, NUM_MODES)
        sources = ica.map_patches_to_sources(inverse_mapping, centred_img_patches)
        source_cov = ica.cov(sources)
        F = ica.err(sources, source_cov)
        
        if i == 0:
            print(' 1st ERROR:', '%.2f'%np.linalg.norm(F,2))
        elif i == 1:
            print(' 2nd ERROR:', '%.2f'%np.linalg.norm(F,2))
        elif i == 2:
            print(' 3rd ERROR:', '%.2f'%np.linalg.norm(F,2))
        else:
            ica.progress(F, i, NUM_ITERATIONS)
            
    print('\n ------------------------------------------')
    print(' * ITERATIONS COMPLETE    \n')
        
########################## SAVE FINAL MODEL ###################################
    
    if SAVE_TO_FILE == 1:
        print(' * SAVING MODEL TO FILE ... ', end='')    
        ica.save_model(SAVE_DIR, inverse_mapping, PATCH_DIM, NUM_MODES)
        print('DONE')
        
######################## LOAD SOLUTION FROM FILE #############################

if LOAD_FROM_FILE == 1:
    print(' * LOADING MODEL FROM FILE ... ', end='')    
    inverse_mapping, PATCH_DIM, NUM_MODES = ica.load_model(LOAD_DIR)
    print('DONE')

#################### APPLY AND VISUALISE SOLUTION #############################

final_img, modal_images = \
    ica.visualise_solution(I, inverse_mapping, PATCH_DIM, NUM_MODES)

if VISUALISE == 1:
    
    # Generate the final greyscale output image, as well as the stack of 
    # modal component images from which it derives
    print(' * GENERATING VISUALISATION ... ', end='')
    print('DONE')

    # Plot the solution
    fig, ax = plt.subplots(figsize=(13,13))
    ax.imshow(final_img)
    

if SAVE_TO_FILE == 1:

    print(' * SAVING VISUALISATION TO FILE ... ', end='')
    ica.save_visualisation(SAVE_DIR, final_img)
    print('DONE')
