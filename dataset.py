import tifffile as tiff
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값
# MAX_PIXEL_VALUE = 255.0 # 이미지 정규화를 위한 픽셀 최대값
def new_dataset(image):
    """ make new image (threshold)"""
    new_image = np.zeros_like(image[:, :, 0])
    for (prob,idx) in [(0.4, 6), (0.4, 5), (0.4, 1), (0.3, 0)]:
        new_image += prob * image[:, :, (idx)]
    return new_image
    
def get_img_762bands(path):
    image = tiff.imread(path)
    #img = image[:, :, (6, 5, 1)]#.astype(np.uint8)
    img = np.float32(image)/MAX_PIXEL_VALUE # 정규화
    img = new_dataset(img)
    #print('processing')
    return np.float32(img) 

def get_mask_arr(path):
    img = tiff.imread(path)
    img = np.float32(img) # 정규화
    return img

# BASE = '/content/gdrive/MyDrive/forest fire/dataset'
BASE = '.'
train_meta = pd.read_csv(f'{BASE}/train_meta.csv').sample(n=10000,random_state=42)
test_meta = pd.read_csv(f'{BASE}/test_meta.csv')

# 데이터 위치
IMAGES_PATH = f'{BASE}/train_img/'
MASKS_PATH = f'{BASE}/train_mask/'

# 가중치 저장 위치
SAVE_PATH = f'{BASE}/train_output/'
MODEL_SAVE = f'{SAVE_PATH}/best_UNet_Base_model.pth'



class CustomDataset(Dataset):
   def __init__(self, imgs_path: list, masks_path: list=None, transform=None, mode='train'):
       self.imgs_path = imgs_path
       self.masks_path = masks_path
       self.transform = transform
       self.mode = mode

   def __len__(self):
       return len(self.imgs_path)

   def __getitem__(self, idx):
       img_path = self.imgs_path[idx]
       img = get_img_762bands(img_path)
       #print(img.shape)
       img = np.reshape(img, (1, 256, 256))

       if self.transform:
           img = self.transform(img)

       if self.mode == 'train':
           mask_path = self.masks_path[idx]
           mask = get_mask_arr(mask_path)

           if self.transform:
               mask = self.transform(mask)

           mask = np.reshape(mask, (1, 256, 256))
           return img, mask

       elif self.mode == 'valid':
           mask_path = self.masks_path[idx]
           mask = get_mask_arr(mask_path)

           if self.transform:
               mask = self.transform(mask)

           mask = np.reshape(mask, (1, 256, 256))
           return img, mask

       else:  # test
           return img