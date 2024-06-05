# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:56:57 2024

@author: shane


### Manuel D'utilisation ###
    
    -Faire en sorte de pip install les package nécessaires
    
    -Pour choisir un slice different => voir ligne 97
    
    -Pour choisir un plan de coup different => voir ligne 98 (pour toujours avoir une image assez grande pour être segmentée utiliser la vue axial: im_data[: , : ,slice_index])
    
    -Sur l'interface :  -Load Image => il faut utiliser un fichier .nii
                        -Sur chacque organe qu'il faut segmenter = > utliser un seed different (ajuster le brush pour des petits organes)
                        -Placer les backround seeds autour des organes 
                        -Cliquer sur segment pour finaliser
                        
    -La segmentation sera sauvegarder dans le même dossier que l'image d'origine (si il y a des problèmes c'est prossible de retirer la fonction save segmented image')
    
"""

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import nibabel as nib
from skimage import exposure
import os

class GrowCutApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GrowCut Segmentation")

        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<B1-Motion>", self.on_paint)

        self.image = None
        self.image_tk = None
        self.seeds = []
        self.labels = None
        self.brush_size = 10

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT)

        self.segment_button = tk.Button(root, text="Segment", command=self.run_segmentation)
        self.segment_button.pack(side=tk.LEFT)

        self.seed1_button = tk.Button(root, text="Seed 1", command=lambda: self.set_brush_type(1))
        self.seed1_button.pack(side=tk.LEFT)

        self.seed2_button = tk.Button(root, text="Seed 2", command=lambda: self.set_brush_type(2))
        self.seed2_button.pack(side=tk.LEFT)

        self.seed3_button = tk.Button(root, text="Seed 3", command=lambda: self.set_brush_type(3))
        self.seed3_button.pack(side=tk.LEFT)

        self.seed4_button = tk.Button(root, text="Seed 4", command=lambda: self.set_brush_type(4))
        self.seed4_button.pack(side=tk.LEFT)

        self.background_button = tk.Button(root, text="Background", command=lambda: self.set_brush_type(0))
        self.background_button.pack(side=tk.LEFT)

        self.brush_size_slider = tk.Scale(root, from_=1, to=10, orient=tk.HORIZONTAL, label="Brush Size", command=self.set_brush_size)
        self.brush_size_slider.pack(side=tk.LEFT)
        self.brush_size_slider.set(5)  # Default brush size

        self.brush_type = 1  # Default to seed 1

    def on_paint(self, event):
        if self.image:
            x, y = event.x, event.y
            for i in range(-self.brush_size // 2, self.brush_size // 2 + 1):
                for j in range(-self.brush_size // 2, self.brush_size // 2 + 1):
                    if 0 <= x + i < self.image.width and 0 <= y + j < self.image.height:
                        self.seeds.append((x + i, y + j, self.brush_type))
                        color = self.get_seed_color(self.brush_type)
                        self.canvas.create_oval(x+i-2, y+j-2, x+i+2, y+j+2, fill=color, outline=color)

    def get_seed_color(self, seed_type):
        colors = {0: "black", 1: "red", 2: "blue", 3: "green", 4: "yellow"}
        return colors.get(seed_type, "red")

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_file_path = file_path  # Store the file path
            img_nib = nib.load(file_path)
            im_data = img_nib.get_fdata()

            slice_index = 240  # Change this index to extract a different slice
            volume = im_data[:, slice_index, :].T

            # Rescale intensity to 8-bit
            min_intensity = -1023
            max_intensity = 1423
            scaled_volume = (volume - min_intensity) * (255 / (max_intensity - min_intensity))

            # Convert the scaled volume to uint8
            volume_rescaled = scaled_volume.astype(np.uint8)

            self.image = Image.fromarray(np.uint8(volume_rescaled), mode='L')  # Convert to uint8 for PIL
            self.image_tk = ImageTk.PhotoImage(self.image)
            self.canvas.config(width=self.image_tk.width(), height=self.image_tk.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
            self.seeds = []
            self.labels = np.zeros((self.image.height, self.image.width), dtype=np.int32)

    def set_brush_type(self, brush_type):
        self.brush_type = brush_type

    def set_brush_size(self, size):
        self.brush_size = int(size)

    def run_segmentation(self):
        # Initialize labels and strength arrays
        height, width = self.image.height, self.image.width
        labels = np.zeros((height, width), dtype=np.int32)
        strengths = np.zeros((height, width), dtype=np.float32)

        # Set seed labels and strengths
        for x, y, seed_type in self.seeds:
            labels[y, x] = seed_type  # No need to add 1 here
            strengths[y, x] = 1.0

        # Convert image to numpy array
        image_np = np.array(self.image)

        # Run the GrowCut algorithm
        labels = self.growcut(image_np, labels, strengths)

        # Create segmentation result
        segmented_image = self.create_segmented_image(labels)

        # Save segmented image
        self.save_segmented_image(segmented_image, self.image_file_path)

        # Display the segmented image
        self.show_segmented_image(segmented_image)

    def growcut(self, image, labels, strengths, max_iter=50):
        height, width = image.shape
        labels_new = labels.copy()
        strengths_new = strengths.copy()

        for _ in range(max_iter):
            changed = False
            
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                                continue
                            if labels[ny, nx] != 0 and labels[ny, nx] != labels[y, x]:
                                gval = self.compute_g(image[y, x], image[ny, nx])
                                new_strength = gval * strengths[ny, nx]
                                if new_strength > strengths_new[y, x]:
                                    strengths_new[y, x] = new_strength
                                    labels_new[y, x] = labels[ny, nx]
                                    changed = True
            labels, strengths = labels_new.copy(), strengths_new.copy()
            if not changed:
                print(f"Max_it =  {_}")
                break
            
            

        return labels

    def compute_g(self, p1, p2):
        return np.exp(-np.linalg.norm(p1 - p2) / 3)

    def create_segmented_image(self, labels):
        segmented_image = Image.new("RGB", (labels.shape[1], labels.shape[0]))
        draw = ImageDraw.Draw(segmented_image)
        for y in range(labels.shape[0]):
            for x in range(labels.shape[1]):
                color = self.get_segment_color(labels[y, x])
                draw.point((x, y), fill=color)
        return segmented_image

    def get_segment_color(self, label):
        colors = {0: "black", 1: "white", 2: "blue", 3: "green", 4: "yellow"}
        return colors.get(label, "black")

    def save_segmented_image(self, segmented_image, original_file_path):
        # Get the directory of the original file
        directory = os.path.dirname(original_file_path)
        # Construct the filename for the segmented image
        segmented_file_name = "Segmented_image.png"
        # Construct the full path for saving the segmented image
        segmented_file_path = os.path.join(directory, segmented_file_name)
        # Save the segmented image
        segmented_image.save(segmented_file_path)
        # Display a message box indicating the successful save
        messagebox.showinfo("Segmented Image Saved", f"The segmented image has been saved at:\n{segmented_file_path}")

    def show_segmented_image(self, segmented_image):
        self.image_tk = ImageTk.PhotoImage(segmented_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)

if __name__ == "__main__":
    root = tk.Tk()
    app = GrowCutApp(root)
    root.mainloop()
