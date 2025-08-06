import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import cv2

# delete background
# zoom in
# min-max normalize + scale (-1,1)
class ImageDataset(Dataset):
    def __init__(self, root):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
       
    def __getitem__(self, index):
        B_path = self.files_B[index % len(self.files_B)]
        A_path = self.files_A[index % len(self.files_A)]
        A_image = np.load(A_path, allow_pickle=True)
        B_image = np.load(B_path, allow_pickle=True)
        
        # Delete background
        # A_image = ImageDataset.delete_background(A_image)
       
        # Random Zoom In
        if np.random.rand() > 0.1:
            A_image = ImageDataset.zoom_in_for_head(A_image)
            B_image = ImageDataset.zoom_in_for_head(B_image)
            
        # Min-max normalization
        A_image = (A_image - np.min(A_image)) / (np.max(A_image) - np.min(A_image))
        B_image = (B_image - np.min(B_image)) / (np.max(B_image) - np.min(B_image))
        
        # Scale to (-1, 1)
        A_image = (A_image * 2.0) - 1.0
        B_image = (B_image * 2.0) - 1.0
        
        A_image = Image.fromarray(A_image)
        B_image = Image.fromarray(B_image)
        A_image = self.transform(A_image)
        B_image = self.transform(B_image)
        
        return {'A': A_image, 'B': B_image}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
   
    @staticmethod
    def delete_background(image, value=0):
        # Fix: use np.amin instead of np.min to avoid recursion
        min_value = np.amin(image)
        image[image == min_value] = value
        return image
    
    @staticmethod
    def zoom_in(image):
        # Zoom in ảnh bằng cách nội suy tuyến tính
        zoom_factor = 1.4
        zoomed = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
        # Lấy kích thước ảnh
        h_image, w_image = image.shape
        # Cắt phần trung tâm để giữ nguyên kích thước
        h, w = zoomed.shape
        start_h = h - h_image - h//10
        start_w = (w - w_image) // 2
        zoomed_cropped = zoomed[start_h:start_h + h_image, start_w:start_w + w_image]
        return zoomed_cropped
    
    @staticmethod
    def zoom_in_for_head(image):
        """
        Zooms in on the center of the image by a random factor between 3 and 3.5.
        image: numpy array of shape (H, W) representing the image.
        """
        # shape of the input image
        h_image, w_image = image.shape

        zoom_factor = np.random.uniform(3, 3.2)
        zoomed_image = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

        # Center crop the zoomed image to the original size
        h, w = zoomed_image.shape
        start_h = (h-h_image)//2
        start_w = (w-w_image)//2
        cropped_image = zoomed_image[start_h:start_h+h_image, start_w:start_w+w_image]
        return cropped_image


class ValDataset(Dataset):
    def __init__(self, root):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        
    def __getitem__(self, index):
        B_path = self.files_B[index % len(self.files_B)]
        A_path = self.files_A[index % len(self.files_A)]
        
        A_image = np.load(A_path, allow_pickle=True)
        B_image = np.load(B_path, allow_pickle=True)
        

        # Delete background
        A_image = ImageDataset.delete_background(A_image)
            
        # Min-max normalization
        A_image = (A_image - np.min(A_image)) / (np.max(A_image) - np.min(A_image))
        B_image = (B_image - np.min(B_image)) / (np.max(B_image) - np.min(B_image))
        
        # Scale to (-1, 1)
        A_image = (A_image * 2.0) - 1.0
        B_image = (B_image * 2.0) - 1.0
        
        A_image = Image.fromarray(A_image)
        B_image = Image.fromarray(B_image)
        A_image = self.transform(A_image)
        B_image = self.transform(B_image)
        
        
        return {'A': A_image, 'B': B_image, 'base_name': os.path.basename(A_path)}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
    @staticmethod
    def delete_background(image, value=0):
        # Fix: use np.amin instead of np.min to avoid recursion
        min_value = np.amin(image)
        image[image == min_value] = value
        return image



# print('test')
# ds = ImageDataset('/home/PET-CT/tiennh/autopet256/train')
# print(len(ds))
# from torch.utils.data import DataLoader

# dl = DataLoader(ds, batch_size=1, shuffle=False)
# for i, data in enumerate(dl):
#     print('main', i)
#     print(data['A'].shape, data['B'].shape)
#     print(data['A'].max(), data['A'].min(), data['B'].max(), data['B'].min())
#     break