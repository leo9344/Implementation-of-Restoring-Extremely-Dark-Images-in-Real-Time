import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from torchvision import models
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import normalized_root_mse as NRMSE
import rawpy
import imageio
from tqdm import tqdm
import torchvision.transforms as tfs
def define_weights(num):
    weights = np.float32((np.logspace(0,num,127, endpoint=True, base=10.0)))
    weights = weights/np.max(weights)
    weights = np.flipud(weights).copy()    
    return weights

# I didnt figure out what 'get_na' means.
# This part is copied from official implementation
def get_na(bins,weights,img_low,amp=5):
    H,W = img_low.shape
    arr = img_low*1
    selection_dict = {weights[0]: (bins[0]<=arr)&(arr<bins[1])}
    for ii in range(1,len(weights)):
        selection_dict[weights[ii]] = (bins[ii]<=arr)&(arr<bins[ii+1])
    mask = np.select(condlist=selection_dict.values(), choicelist=selection_dict.keys())
   
    mask_sum1 = np.sum(mask,dtype=np.float64)
    
    na1 = np.float32(np.float64(mask_sum1*0.01*amp)/np.sum(img_low*mask,dtype=np.float64))
# As in SID max amplification is limited to 300
    if na1>300.0:
        na1 = np.float32(300.0)
    if na1<1.0:
        na1 = np.float32(1.0)
    
    selection_dict.clear()

    return na1

def get_sid_amp(img_low, amp=5):
    H, W = img_low.shape
    return amp*(H*W)/(img_low.sum())

def get_tgt_and_low(root_path, list_path, amp_file_path, debug, debug_iter):
    # list_path = "F:/Learning-to-see-in-the-dark/Sony/Sony_train_list.txt" 
    # root_path = "F:/Learning-to-see-in-the-dark/Sony"
    # amp_file_path = "./checkpoint/train_amp.txt"
    list_file = open(list_path, "r")
    list_lines = list_file.readlines()
    if debug:
        list_lines = list_lines[:debug_iter]
    low_list = []
    tgt_list = []

    mean = 0

    # bins = np.float32((np.logspace(0,8,128, endpoint=True, base=2.0)-1))/255.0
    # print('\nEdges:{}, dtype:{}\n'.format(bins,bins.dtype))#, file = open(amp_file_path, 'w'))
    # weights5 = define_weights(5)
    # print('------- weights: {}\n'.format(weights5))#, file = open(amp_file_path, 'w'))

    for line in tqdm(list_lines, desc=f"Loading from {list_path}"):
        low_path, tgt_path, iso, focal = line.split(" ")
        low_path = os.path.join(root_path, low_path)
        tgt_path = os.path.join(root_path, tgt_path)
        # rawpy.postprocess Args:
        # use_camera_wb (bool) – whether to use the as-shot white balance values
        # half_size (bool) – outputs image in half size by reducing each 2x2 block to one pixel instead of interpolating
        # no_auto_bright (bool) – whether to disable automatic increase of brightness
        # output_bps (int) – 8 or 16

        # Loading target image #
        raw = rawpy.imread(tgt_path)
        img_tgt = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16).copy()
        raw.close()
        img_tgt=np.float32(img_tgt/65535.0)
        h, w, channel = img_tgt.shape

        # Correction h,w -> 32*n
        corrected_flag = False
        if h % 32 != 0:
            print("Correcting height.")
            h = h - (h % 32)
            img_tgt = img_tgt[:h, :, :]
            corrected_flag = True
        
        if w % 32 != 0:
            print("Correcting width.")
            w = w - (w % 32)
            img_tgt = img_tgt[:, :w, :]
            corrected_flag = True
        
        

        # Loading train images #

        # raw.raw_image_visible 
        # Like raw_image but without margin.
        # Return type: ndarray of shape (hv,wv[,c])
        raw = rawpy.imread(low_path)
        img_low = raw.raw_image_visible.astype(np.float32).copy()
        raw.close()

        if corrected_flag:
            img_low = img_low[:h, :w]

        img_low = (np.maximum(img_low-512, 0) / (16383 - 512)) # idk 16383==Magic number?

        na5 = get_sid_amp(img_low, amp=0.05)#get_na(bins, weights5, img_low)

        # Different from official code, we do not have GT amp.
        H,W = img_low.shape    
        a = np.float32(np.float64(H*W*0.01)/np.sum(img_low,dtype=np.float64))
        

        img_low = (img_low*na5)
        # print('...using na5 : {}'.format(tgt_path[-17:]), file = amp_file_path)
        
        # Adding images
        tgt_list.append(torch.from_numpy(np.transpose(img_tgt, [2, 0, 1])).float())
        low_list.append(torch.from_numpy(img_low))
        
        mean += np.mean(img_low[0::2,1::2],dtype=np.float32)

        
        # print('Image {} base_amp: {}, gt_amp: {}, Our_Amp:{}'.format(i+1,a,ta,na5), file = file_line)

    print('Files loaded : {}/{}, channel mean: {}'.format(len(low_list), len(tgt_list), mean/len(low_list)))
    list_file.close()
    return tgt_list, low_list
        


class sid_dataset(Dataset):
    def __init__(self, cfg, split, training):
        super(sid_dataset, self).__init__()

        assert split in ['train', 'valid', 'test'], "Split must in 'train', 'valid' or 'test'! "
        self.split = split
        self.training = training
        self.sony_img_path = cfg.sony_img_path
        self.debug = cfg.debug
        self.debug_iter = cfg.debug_iter
        self.debug_size = cfg.debug_size

        if self.split == 'train':
            self.list_path = cfg.train_list_path
            self.amp_path = "./checkpoint/train_amp.txt"

        elif self.split == 'valid':
            self.list_path = cfg.valid_list_path
            self.amp_path = "./checkpoint/valid_amp.txt"

        elif self.split == 'test':
            self.list_path = cfg.test_list_path
            self.amp_path = "./checkpoint/test_amp.txt"

        else:
            raise NotImplementedError("???")
        
        self.debug = cfg.debug
        if self.debug == True:
            self.max_img = 100
            
        print(f"Loading {split} set...")
        self.tgt_list, self.low_list = get_tgt_and_low(self.sony_img_path, self.list_path, self.amp_path, self.debug, self.debug_size)
        print(f"{split} set loaded. Total target img num: {len(self.tgt_list)}, Total low img num: {len(self.low_list)}")

        self.aug = self.get_data_aug()
    def __len__(self):
        return len(self.tgt_list)

    @staticmethod
    def get_data_aug():

        return tfs.Compose([
            tfs.RandomVerticalFlip(p=0.5),
            tfs.RandomHorizontalFlip(p=0.2),
            tfs.RandomCrop((512, 512)),
        ])


    def __getitem__(self, index):

        img_tgt = self.tgt_list[index]
        img_low = self.low_list[index]
        
        H,W = img_low.shape
        
        if self.training:
            img_tgt = self.aug(img_tgt)
            img_low = self.aug(img_low)

        else:
            img_low = img_low
            img_tgt = img_tgt
            
        # gt = torch.from_numpy((np.transpose(img_gt, [2, 0, 1]))).float() # C x H x W
        # low = torch.from_numpy(img_low).float().unsqueeze(0)            
        
        return img_tgt, img_low.unsqueeze(0)
        
