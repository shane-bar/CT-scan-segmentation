# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:59:28 2024

@author: matta
"""

import seg_metrics.seg_metrics as sg
import numpy as np
import SimpleITK as sitk
from PIL import Image
from skimage import  exposure 
from skimage.transform import resize
from skimage.filters import threshold_otsu

import matplotlib.pyplot as plt

def comparaison ( seg_path , pred_seg_path , slice_index , metric, labels = [0, 1]  ):
    '''
    

    Parameters
    ----------
    seg_path : string
        true segmentation.
    pred_seg_path : string
        our result.
    slice_index : int
        which slice are we comparing the segmentation on?.
    metrics: list
        which metrics do we want to measure? (dice, and )
    labels : list, optional
        if there are multiple labels ("colours") to compare . The default is [0, 1] for black and white or boolean.

    Returns
    -------
    
    (It prints the dice coefficient)

    
    -------
    dice_number : List
        list of dice coefficient for each label, by default since there is only one label (0 being the background) , it's a list of one float number.

    '''
    
    
    ###CHARGER L'IMAGE GROUND TRUTH
    itk_image = sitk.ReadImage(seg_path)#on lis l'image en tant que SimpleITK.Image
    image_ground_truth_data = sitk.GetArrayFromImage(itk_image)[slice_index,:,:]
    image_uint8_gt =image_ground_truth_data.astype(np.uint8) #on transforme l'image 16 bit en 8 bit
        

    ###CHARGER L'IMAGE SEGMENTEE
    pred_img = np.array(Image.open(pred_seg_path).convert('L'))
    
    image_rescaled_pred = exposure.rescale_intensity(pred_img, in_range='image', out_range=np.uint8) #pas nécessaire et ne change rien car l'image ici est déjà en 8bit

                                #on met le codage de l'image en 8 bit (not necessary) et on reshape l'image à la même shape que l'image groungtruth
    image_uint8_pred = resize(image_rescaled_pred.astype(np.uint8) , np.shape(image_uint8_gt), anti_aliasing=True)
    
    
    ###CONVERSION DES DEUX IMAGES EN BOLLEAN
    #on fait un seuillage
    image_uint8_pred = image_uint8_pred > threshold_otsu(image_uint8_pred)#on choisit otsu arbitrairement pour avoir un seuillage automatique, nothing specific
    image_uint8_gt = image_uint8_gt > threshold_otsu(image_uint8_gt)
    
    
    ###ON VEUT COMPARER LES DEUX IMAGES
    
    gdth_img = image_uint8_gt #l'image avec laquelle nous voulons comparer nos résultats, la réalité: GROUND TRUTH
    pred_img = image_uint8_pred #nos résulats de la segmentation que nous voulons comparer avec la réalité
    csv_file = 'metrics.csv' #nom du fichier csv que nous créerons en mettant nos données (nos metrics)
    
    metrics = sg.write_metrics(labels=labels[1:],  # exclude background
                               gdth_img=gdth_img, #ground truth
                               pred_img=pred_img, #image à comparer
                               csv_file=csv_file, #nom du fichier contenant metrics créé(metrics est un dictionnaire de listes)
                               metrics = metric #on utilise la méthode dice metrics est alors une liste d'un dictionnaire avec dans ce dictionnaire une clé pour une lsite de label et des clés pour qhaque coefficient correspondan tà une méthode, ici dice
                               )
    print(metrics)
    print('')
    
    dice_number = metrics[0]['dice'] #liste des coefficient de dice
    msd = metrics[0]['msd'][0]
    titre = "Dice index: "
    for i in range (len(labels)-1): 
        print(f"L'indice de dice pour le label {labels[i+1]} est de {np.round(metrics[0]['dice'][i], 3)}")
        titre+= str(np.round(metrics[0]['dice'][i], 3))
        titre += " pour le label "
        titre += str(labels[i+1]) 
        if np.round(metrics[0]['dice'][0], 3) == 1:
            print("Les images sur ce label sont identiques")
    titre += " Mean surface index: "
    titre+= str(np.round(msd, 3))
            
    ###AFFICHAGE DES DEUX IMAGES
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(pred_img, cmap='gray')
    plt.title("Image segmenté par notre algorithme")
    plt.subplot(1,2,2)
    plt.imshow(gdth_img, cmap='gray')
    plt.title("Image segmenté réelle")
    plt.suptitle(titre)

    return dice_number , gdth_img , pred_img, msd 




#chemin des fichiers à lire
#seg_path ='C:/Users/matta/Documents/cours/2A/P2I8/projet/AbdomenCT-1K/AbdomenCT-1K/segmentations/Case_00056.nii.gz'
#pred_seg_path = 'C:/Users/matta/Documents/cours/2A/P2I8/projet/results/segmented_image1.png'  



pred_seg_path  ="D:/projet/results/wassim/reins2.png"
gdth_path = "D:/projet/AbdomenCT-1K/AbdomenCT-1K/segmentations/Case_00209.nii/Case_00209.nii"


metric = ['dice' , 'msd']
slice_index = 240
dice_number , gdth_img , pred_img, mean = comparaison(gdth_path, pred_seg_path, slice_index, metric)
    