# -*- coding: utf-8 -*-
"""
Created on Tue May 21 08:08:39 2024

@author: thiru
"""

import nibabel as nib
import matplotlib.pyplot as plt
from skimage import segmentation, exposure,io
import numpy as np 
from skimage import img_as_ubyte
import matplotlib.colors as mcolors
from scipy.ndimage import rotate



#load the image
file_path = 'C:/Users/thiru/INSA - FIMI/2A/P2I/projet/IRM/AbdomenCT-1K/images/Case_00209_0000.nii/Case_00209_0000.nii'
image = nib.load(file_path)
#get the image data in the form of a list
image_data = image.get_fdata()
# rescale the values from -1024 to ___ (16bits) to 0 to 255 (8bits)
# only the values changed but image still encoded with 16bits 
# because the flood_fill function doesn't work with that
image_normalized =((image_data - image_data.min())*255) / (image_data.max() - image_data.min()) 
# Rescale the image using pre existing function
    #image_normalized = exposure.rescale_intensity(image_data, in_range='image', out_range=np.uint8)

# Convert the rescaled image to uint8 
image_uint8 = image_normalized.astype(np.uint8)

# retransforer en image nii
# new_image = nib.Nifti1Image(image_uint8, image.affine)
# new_image_data = new_image.get_fdata()
# slice_0 = image_data[:, :, 0]  # Extraire la 15è tranche
slice_1 = rotate(image_uint8[:, 240, :], angle = -180).T  # Extraire la première trancherotate(displayed_slice, angle = -180).T

#plt.figure()
#plt.imshow(slice_1.T,cmap='gray')

# seed_point = (100,150,15)
# new_value = 0
# filled_image = segmentation.flood_fill(image_uint8, seed_point, new_value)

# plt.figure()
# plt.imshow(filled_image, cmap='gray')


#########Floodfill
fig, ax = plt.subplots()
# slice_index_y =   # Middle slice for 3D image
# slice_index_x = 
slice_index_z = 54

ax.imshow(slice_1, cmap='gray')
ax.set_title('Click to select seed point')
seed_point = None

label_organ =4
label_backround = 0
def segmentation_zone(event):
    new_value = 120  # The new value to fill with
    print(f"Clicked at: x={event.xdata}, y={event.ydata}")
    milieu = (int(event.ydata), int(event.xdata))  # Note: z-index is included
    print(milieu)
    if milieu is not None:
        print(f"Center point selected: {milieu}")
        distancey = 35
        distancex = 60
        
        image_restricted = slice_1[milieu[0]-distancex:milieu[0]+distancex,milieu[1]-distancey:milieu[1]+distancey]
        
        seed_point = distancex,distancey
        tolerance = 5
        filled_image = segmentation.flood_fill(image_restricted, seed_point, new_value, tolerance=tolerance, connectivity=4) 
        segmented = np.zeros(np.shape(slice_1))
        mask = filled_image == new_value
        segmented[milieu[0]-distancex:milieu[0]+distancex,milieu[1]-distancey:milieu[1]+distancey][mask] = label_organ
        segmented = rotate(segmented, angle = -180).T
        segmented_image_nii = nib.Nifti1Image(segmented, image.affine)
                #segmented_image_nii_data = segmented_image_nii.get_fdata()
        # segmented_image_nii.set_filename('segmented_pancreas2')
        # nib.save(segmented_image_nii, 'segmented_pancreas2')
        # print(f'saved as {segmented_image_nii.get_filename()}')
                # Display the original and filled image
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        ax = axes.ravel()

        ax[0].imshow(slice_1, cmap='gray')
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        ax[1].imshow(image_restricted, cmap='gray')
        ax[1].set_title('restricted zone for segmentation')
        ax[1].axis('off')
        
        # ax[2].imshow(segmented)
        # ax[2].set_title('segmented image')
        # ax[2].axis('off')
        
        # # Define a custom colormap to highlight the segmented region
        # cmap = plt.cm.get_cmap("gray").copy()
        # cmap.set_under(color='black')  # Color for values below 1
        
        # # Display the segmented image
        # ax[2].imshow(segmented, cmap=cmap, vmin=0, vmax=1)
        # ax[2].set_title('Segmented Image')
        # ax[2].axis('off')
        
        # Create a custom colormap
        cmap = mcolors.ListedColormap(['black', 'red'])
        bounds = [0, 0.5, 1]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # Display the segmented image
        ax[2].imshow(segmented, cmap=cmap, norm=norm)
        ax[2].set_title('Segmented Image')
        ax[2].axis('off')
        plt.show()


fig.canvas.mpl_connect('button_press_event', segmentation_zone)
plt.show()


print(image_data.max())