import numpy as np
from skimage.segmentation import random_walker
from skimage.data import binary_blobs
import matplotlib.pyplot as plt
from skimage.io import imsave
import nibabel as nib
import SimpleITK as sitk
from skimage.transform import resize
from skimage.util import invert

file_path = 'Case_00096_0001.nii'
image = nib.load(file_path)
image_data = image.get_fdata()
print(image.get_data_dtype())
# rescale the values from 16bits (-1024 to ___) to 8bits (0 to 255)
image_normalized =((image_data - image_data.min())*255) / (image_data.max() - image_data.min())
# Rescale the image intensities from 16-bit to 8-bit
#image_rescaled = exposure.rescale_intensity(image_data, in_range='image', out_range=np.uint8)

# Convert the rescaled image to uint8
image_uint8 = image_normalized.astype(np.uint8)

new_image = nib.Nifti1Image(image_uint8, image.affine)
new_image_data = new_image.get_fdata()
print(new_image.get_data_dtype())

y = 190
slice_1 = new_image_data[:, y, :].T # Extraire la tranche selon y
slice_1 = resize(slice_1, (512,512) , preserve_range=True)

slice_1 = np.flipud(slice_1)
# Generate synthetic data
data = slice_1

# Create an array of the same shape as data for labels
labels = np.zeros_like(data)
# Définir les couleurs pour chaque région
colors = [(0, 0, 0),        # noir pour le fond
          (1, 0, 0),        # rouge pour le foie
          (0, 0, 1),        # bleu pour la rate
          (0, 1, 0)]        # vert pour les reins
# Créer une carte de couleurs personnalisée
from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(colors)


# Create global variables to store the number of clicks and their positions
n_clicks = 0
clicks = []

# Define the click event handler    
def onclick(event):
    
    global n_clicks, clicks, labels
    if event.xdata is not None and event.ydata is not None:
        x, z= int(event.ydata), int(event.xdata)
        clicks.append((x, z))
        n_clicks += 1
        
        # Assign labels based on the number of clicks
        if n_clicks <= 5:
            labels[x, z] = 1  # Foreground marker
            ax[1].scatter(z, x, c='r', s=50, label='foie')
        elif n_clicks <= 8:
            labels[x, z] = 2  # Foreground marker
            ax[1].scatter(z, x, c='b', s=50, label='rate')
        elif n_clicks <=    18:
            labels[x, z] = 3  # Foreground marker
            ax[1].scatter(z, x, c='g', s=50, label='reins')
        elif n_clicks <= 39:
            labels[x, z] = 4 # Background marker
            ax[1].scatter(z, x, c='purple', s=50, label='Background')
            #fig.canvas.mpl_disconnect(cid)  # Disconnect after two clicks
        elif n_clicks == 40:
            labels[x,z] = 4 # Background marker
            ax[1].scatter(z, x, c='purple', s=50, label='Background')    
            # Perform segmentation using random walker after selecting the labels
            segmented_data = random_walker(data, labels, beta=430, copy=True)
            segmented_data = invert(segmented_data)
            imsave("segmentation.png", segmented_data)
            # Plot the result
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))
            axes[0].imshow(data, cmap='gray')
            axes[0].set_title('Original Data')
            # axes[1].imshow(labels, cmap='gray', interpolation='none')
            # axes[1].set_title('Labels')
            axes[1].imshow(segmented_data,  cmap=custom_cmap)
            axes[1].set_title('Segmented Data')
            plt.show()
        # Redraw the figure
        ax[1].imshow(labels, cmap='gray', interpolation='none')
        plt.draw()

# Plot the initial data
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(data, cmap='gray')
ax[0].set_title('Original Data')


labels = resize(labels, (512,512) , preserve_range=True)
ax[1].imshow(labels, cmap=custom_cmap, interpolation='none')
ax[1].set_title('Labels (click to set)')

# Connect the click event to the handler
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
