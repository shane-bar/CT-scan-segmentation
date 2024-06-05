# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 02:38:02 2024

@author: Kananila
"""
"""
Étapes pour faire tourner le programme: 
Avant de lancer:
    - Modifier les chemins d’accès sur la ligne 232 du code 
Après lancement:
    - Donner les 3 derniers chiffres du numéro de cas voulu
    - Appuyer sur load image
    - Sélectionner le vu de visualisation
    - Scroller en utilisant le 1er curseur 
    - Dès que vous appuyez sur un organe sur l’image original, vous pouvez avoir accès à la segmentation de ce dernier sur la deuxième partie de la fenêtre intitulée “Segmented image”. La segmentation en 3D est observable en utilisant le deuxième curseur à droite.
    - La segmentation s’actualise à chaque fois qu’un clic a eu lieu sur l’image original
    - La taille de la zone 3D à segmenter peut être modifiée manuellement sur l’interface mais la taille par défaut correspond à celle du foie.
    - En fonction de la segmentation effectuée, modifier la tolérance.
    pour enregistrer une segmentation, lui donner un nom et appuyer sur “save segmentation”
Conseils d’utilisation:
    - Cliquer au milieu de l’organe à segmenter.
La vue axiale ne semble pas marcher

"""
import tkinter as tk
from tkinter import scrolledtext
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import segmentation, exposure,io, img_as_ubyte
import numpy as np 
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np


class FloodFill(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Flood Fill segmentation")
     
        # Create a frame for original image
        self.orig_frame = tk.Frame()
        self.orig_frame.pack(side=tk.LEFT, padx=5, pady=5)
        self.orig_label = tk.Label(self.orig_frame, text="Original Image")
        self.orig_label.pack()
        self.orig_canvas = tk.Canvas(self.orig_frame)
        self.orig_canvas.pack()

        # Create a frame for segmented image
        self.segment_frame = tk.Frame()
        self.segment_frame.pack(side=tk.LEFT, padx=5, pady=5)
        self.segment_label = tk.Label(self.segment_frame, text="Segmented Image")
        self.segment_label.pack()
        self.segment_canvas = tk.Canvas(self.segment_frame)
        self.segment_canvas.pack()
     
        # Scroll original image in 3D 
        self.segment_button_1 = tk.Label(text="scroll original image")
        self.segment_button_1.pack()
        self.index = tk.Scale(self, from_=1, to=512, orient='horizontal', command = self.bind_scale_orig)
        self.index.pack()
        
        # Scroll segmented image in 3D 
        self.segment_button_2 = tk.Label(text="scroll segmented image")
        self.segment_button_2.pack()
        self.index_segm = tk.Scale(self, from_=1, to=511, orient='horizontal', command = self.bind_scale_segm)
        self.index_segm.pack()
        
        # adaptation in another computer
        # select the case and put it on the frame
        # self.indication = tk.Label(text="Give the entire path in the 2nd box")
        # self.indication.pack()
        
        self.case_number = tk.Entry()
        self.case_number.insert(0, "Last 3 digits of the case")
        self.case_number.pack()
        
        # DOESN'T work so if you work on an other computer, you'll have to change a bit the load_image function
        # self.file_path = tk.Entry()                   
        # self.file_path.insert(0, "entire path")
        # self.file_path.pack()
        
        self.bouton1 = tk.Button(text="load image")
        self.bouton1.bind('<Button-1>',self.load_image)
        self.bouton1.pack()
        
        self.image_uint8 = None
        
        # choose the preferred axis
        self.axis = tk.StringVar()  # Variable commune aux 2 RadioButtons
        self.axis.set("axis")
        
        self.profile = tk.Radiobutton(text="profile", variable=self.axis, value="profile")
        self.profile.pack()
        
        self.coronal = tk.Radiobutton(text="coronal", variable=self.axis, value="coronal")
        self.coronal.pack()
        
        self.axiale = tk.Radiobutton(text="axiale", variable=self.axis, value="axiale")
        self.axiale.pack()
    
        self.bouton = tk.Button(text="select")
        self.bouton.bind('<Button-1>',self.select_axis)
        self.bouton.pack()
        
        #tailles de la zone restrainte pour la segmentation
        self.distance_x = 80 #hauteur en coronal pour une image non transposé / largeur en réalité
        self.distance_y = 60 #corresponds au profondeur en coronal
        self.distance_z = 30
        
        self.distance_x_entry = tk.Entry()
        self.distance_x_entry.insert(0, "distance_x")
        self.distance_x_entry.pack()
        
        self.distance_y_entry = tk.Entry()
        self.distance_y_entry.insert(0, "distance_y")
        self.distance_y_entry.pack()
        
        self.distance_z_entry = tk.Entry()
        self.distance_z_entry.insert(0, "distance_z")
        self.distance_z_entry.pack()
        
        self.bouton_distance = tk.Button(text="select distance")
        self.bouton_distance.bind('<Button-1>',self.get_distance)
        self.bouton_distance.pack()
        
        #tolérance de floodfill algorithme
        self.tolerance = 3
        self.tolerance_ = tk.Entry()
        self.tolerance_.insert(0, "tolerance")
        self.tolerance_.pack()
        
        self.bouton_tolerance = tk.Button(text="apply tolerance")
        self.bouton_tolerance.bind('<Button-1>',self.get_tolerance)
        self.bouton_tolerance.pack()
                
        self.segment_button = tk.Label(text="Click on the organ on the original image to segment")
        #self.segment_button.bind('<Button-1>', self.flood_fill_segmentation)
        self.segment_button.pack(side=tk.TOP, pady=5)
        
        #initialisation de tous les variables
        self.seed_point = None

        self.displayed_slice = None
        self.image_tk = None
        self.segm_image_tk = None
        self.displayed_segm_slice = None
        self.segmented_image = None
        self.original_image = None
        self.image_tk = None
        self.segm_image_tk = None
        self.seeds = []
        self.labels = None

        self.orig_canvas.bind("<Button-1>", self.flood_fill_segmentation)
        
        #bouton pour enregistrer
        self.save_as = tk.Entry()
        self.save_as.insert(0, "save name")
        self.save_as.pack()
                
        self.bouton_save = tk.Button(text="save segmentation")
        self.bouton_save.bind('<Button-1>',self.save_segmentation)
        self.bouton_save.pack(side=tk.LEFT)


    def flood_fill_segmentation(self,event):
        """
        

        Parameters
        ----------
        event : un clic gauche ou droit sur la zone de l'image à segmenter
            Il est relié au canvas de la zone de l'image originale'

        Returns
        -------
        Il actualise la segmentation à chaque clic.

        """
        index_choisi = self.index.get()
        print(f"Clicked at: x={event.y}, y={event.x}")
        milieu = (int(event.y), int(event.x))  # Note: z-index is included
        print(milieu)
        index_choisi = self.index.get()
        self.segmented_image = self.image_uint8.copy()
        tolerance = self.tolerance
        if milieu is not None:
            print(f"Center point selected: {milieu}")
            distance_x = self.distance_x #hauteur en coronal pour une image non transposé / largeur en réalité
            distance_y = self.distance_y #corresponds au profondeur en coronal
            distance_z = self.distance_z
            seed_point = distance_x, distance_y, distance_z
            new_value = 255  # The new value to fill with

            if self.axis.get() == "profile":
                image_restricted = self.image_uint8[index_choisi-distance_x :index_choisi+distance_x,milieu[0]-distance_y:milieu[0]+distance_y,milieu[1]-distance_z:milieu[1]+distance_z]
                filled_image = segmentation.flood_fill(image_restricted, seed_point, new_value, tolerance = self.tolerance)
                self.segmented_image[index_choisi-distance_x :index_choisi+distance_x,milieu[0]-distance_y:milieu[0]+distance_y,milieu[1]-distance_z:milieu[1]+distance_z] = filled_image
                
            if self.axis.get() == "coronal":
                image_restricted = self.image_uint8[milieu[0]-distance_x:milieu[0]+distance_x, index_choisi-distance_y :index_choisi+distance_y ,milieu[1]-distance_z:milieu[1]+distance_z]
                filled_image = segmentation.flood_fill(image_restricted, seed_point, new_value, tolerance = self.tolerance)
                self.segmented_image[milieu[0]-distance_x:milieu[0]+distance_x, index_choisi-distance_y :index_choisi+distance_y ,milieu[1]-distance_z:milieu[1]+distance_z] = filled_image
                
            if self.axis.get() == "axial":
                image_restricted = self.image_uint8[milieu[0]-distance_x:milieu[0]+distance_x,milieu[1]-distance_y:milieu[1]+distance_y, index_choisi-distance_z :index_choisi+distance_z]
                filled_image = segmentation.flood_fill(image_restricted, seed_point, new_value, tolerance = self.tolerance)
                self.segmented_image[milieu[0]-distance_x:milieu[0]+distance_x,milieu[1]-distance_y:milieu[1]+distance_y, index_choisi-distance_z :index_choisi+distance_z] = filled_image
                
            # self.segmented_image = segmented_image

    # load image    
    def load_image(self, event):
            """
        

        Parameters
        ----------
        event : Clic sur le bouton "load image"
        
        !!!! Le chemin est adapté à un ordi, il faudra adapter le case_file_path pour que le numéro de cas sufisse à avoir accès à l'image'
        Returns
        -------
        Actualise l'image à segmenter selectionné.

        """
            if self.case_number.get() != None:
                case = self.case_number.get()
                print(case)
                case_file_path = f"C:/Users/thiru/INSA - FIMI/2A/P2I/projet/IRM/AbdomenCT-1K/images/Case_00{case}_0000.nii/Case_00{case}_0000.nii"
            # elif self.file_path.get() != None:
            #     case_file_path = f"{self.file_path.get()}"
                
            print(case_file_path)
            self.case_file_path = case_file_path
            
            image = nib.load(self.case_file_path)
            #get the image data in the form of a list
            image_data = image.get_fdata()
            # rescale the values from -1024 to ___ (16bits) to 0 to 255 (8bits)
            # only the values changed but image still encoded with 16bits 
            image_normalized = exposure.rescale_intensity(image_data, in_range='image', out_range=np.uint8)
            print(np.size(image_normalized))
            # Convert the rescaled image to uint8 
            self.image_uint8 = image_normalized.astype(np.uint8)
            self.original_image = image

    def save_segmentation(self, event):
        """
    

        Parameters
        ----------
        event : Clic sur le bouton "save segmentation"
        
        Returns
        -------
        enregistre l'image segmentée sur le bureau de l'utilisateur.
    
        """
        # retransforer en image nii
        case = self.case_number.get()
        segmented_image_nii = nib.Nifti1Image(self.segmented_image, self.original_image.affine)
        #segmented_image_nii_data = segmented_image_nii.get_fdata()
        name = str(self.save_as.get())
        segmented_image_nii.set_filename(f'{name}.nii')
        nib.save(segmented_image_nii, f'{name}.nii')
        print(f'saved as {segmented_image_nii.get_filename()}')
       
    def get_distance(self, event):
        """ prend les distances données par l'utilisateur sur l'interface"""
        self.distance_x = int(self.distance_x_entry.get())
        self.distance_y = int(self.distance_y_entry.get())
        self.distance_z = int(self.distance_z_entry.get())
    
    def get_tolerance(self, event):
        """ prend la tolérance à prendre en compte donnée par l'utilisateur sur l'interface"""
        self.tolerance = int(self.tolerance_.get())
        
    def select_axis(self, event):
            """permet de sélectionner la coupe voulu"""
            if self.axis.get() == "profile":
                #le scale a autant de chiffres que la taile en x
                self.slice = 'x'                
                size = np.size(self.image_uint8[:,1,1])
                maxi = size - 1
            elif self.axis.get() == "coronal":
                self.slice = 'y'
                #le scale a autant de chiffres que la taile en x
                size = np.size(self.image_uint8[1,:,1])
                maxi = size - 1
            elif self.axis.get() == "axial":
                self.slice = 'z'
                size = np.size(self.image_uint8[1,1,:])
            #     maxi = size - 1 # get the number on the scaler
            # self.maxi = maxi
            # print(maxi)
              
    def bind_scale_segm(self,event):
            """
            Parameters
            ----------
            event : déplacement de curseur
    
            Returns
            -------
            pemet d'associer le numéro affiché sur le curseur à l'index de coupe de l'image voulu pour l'image segmentée: c'est le 2è curseur
    
            """
            slice_index = self.index_segm.get()
            
            if self.axis.get() == "profile":
                displayed_slice = self.segmented_image[slice_index, : , :]
                
            elif self.axis.get() == "coronal":
                displayed_slice = self.segmented_image[:, slice_index, :]
                
            elif self.axis.get() == "axial":
                displayed_slice = self.segmented_image[:, : , slice_index]
                
            self.displayed_slice_segm = displayed_slice
            # put the image that automatically depends on the nb of the scaler
            
            self.displayed_slice_segm_tk = Image.fromarray(np.uint8(self.displayed_slice_segm), mode='L')
            self.image_segm_tk = ImageTk.PhotoImage(self.displayed_slice_segm_tk)
            self.segm_image_tk = ImageTk.PhotoImage(self.displayed_slice_segm_tk)
            self.segment_canvas.config(width=self.image_segm_tk.width(), height=self.image_segm_tk.height())
            self.segment_canvas.create_image(0, 0, anchor=tk.NW, image=self.segm_image_tk)
            self.labels = np.zeros((self.displayed_slice_segm_tk.height, self.displayed_slice_segm_tk.width), dtype=np.int32)
        
    def bind_scale_orig(self,event):
            """
            Parameters
            ----------
            event : déplacement de curseur
    
            Returns
            -------
            pemet d'associer le numéro affiché sur le curseur à l'index de coupe de l'image voulu pour l'image originale: 1er curseur
    
            """
            slice_index = self.index.get()
            
            if self.axis.get() == "profile":
                displayed_slice = self.image_uint8[slice_index, : , :]
                
            elif self.axis.get() == "coronal":
                displayed_slice = self.image_uint8[:, slice_index, :]
                
            elif self.axis.get() == "axial":
                displayed_slice = self.image_uint8[:, : , slice_index]
                
            self.displayed_slice = displayed_slice
            # put the image that automatically depends on the nb of the scaler
            
            self.displayed_slice_tk = Image.fromarray(np.uint8(self.displayed_slice), mode='L')
            self.image_tk = ImageTk.PhotoImage(self.displayed_slice_tk)
            self.orig_image_tk = ImageTk.PhotoImage(self.displayed_slice_tk)
            self.orig_canvas.config(width=self.image_tk.width(), height=self.image_tk.height())
            self.orig_canvas.create_image(0, 0, anchor=tk.NW, image=self.orig_image_tk)
            self.labels = np.zeros((self.displayed_slice_tk.height, self.displayed_slice_tk.width), dtype=np.int32)

if __name__ == "__main__":
    app = FloodFill()
    app.mainloop()