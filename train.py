import numpy as np
import os
import shutil
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import glob
import torch.nn as nn
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import imageio
from torchsummary import summary

from dataloader import sid_dataset
from config import cfg
from model import Net
from utils import MS_SSIM

base_lr = cfg.base_lr
reduce_lr_by = cfg.reduce_lr_by
atWhichReduce = cfg.atWhichReduce
batch_size = cfg.batch_size
atWhichSave = cfg.atWhichSave
max_iter = cfg.max_iter
debug = cfg.debug
debug_iter = cfg.debug_iter
weight_save_path = cfg.weight_save_path
csv_save_path = cfg.csv_save_path
img_save_path = cfg.img_save_path
val_interval = 100

device = 'cuda' if torch.cuda.is_available() else "cpu"

train_dataset = sid_dataset(cfg, "train", training=True)
valid_dataset = sid_dataset(cfg, 'valid', training=True)
# test_dataset = sid_dataset(cfg, 'test', training=False)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = Net().to(device)

summary(model, (1,512,512), batch_size=1, device=device)

L1_loss = torch.nn.L1Loss()
MS_SSIM_loss = MS_SSIM(data_range=1.0, size_average=True, channel=3)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.base_lr)

# --- Validation --- #
def valid(model, dataloader, epoch, image_save_path, metric_save_path):
    model.eval()
    loss_per_epoch = 0
    l1loss_per_epoch = 0
    msssim_per_epoch = 0
    psnr_ls = []
    ssim_ls = []

    for idx, data in tqdm(enumerate(dataloader), desc=f"Valid Epoch: {epoch}", total=round(len(valid_dataset)/batch_size)):
        tgt, low = data
        tgt = tgt.to(device)
        low = low.to(device)
        pred = model(low)
        l1_loss = L1_loss(pred, tgt)
        ms_ssim = 1 - MS_SSIM_loss(pred, tgt) #? why 1-msssim

        loss_total = 0.8 * l1_loss + 0.2 * ms_ssim
        loss_per_epoch += loss_total.item()
        l1loss_per_epoch += l1_loss.item()
        msssim_per_epoch += ms_ssim.item()

        pred = (np.clip(pred[0].detach().cpu().numpy().transpose(1,2,0),0,1)*255).astype(np.uint8)
        tgt = (np.clip(tgt[0].detach().cpu().numpy().transpose(1,2,0),0,1)*255).astype(np.uint8)
        psnr_img = PSNR(pred,tgt)
        ssim_img = SSIM(pred,tgt,multichannel=True)
        
        psnr_ls.append(psnr_img)
        ssim_ls.append(ssim_img)

        if idx in [0,1,2,3,7,10,11,12,13,19,20,30,35,41,46,47,48]:
            imageio.imwrite(os.path.join(image_save_path,'{}_{}_gt.jpg'.format(idx,epoch)), tgt)
            imageio.imwrite(os.path.join(image_save_path,'{}_{}_psnr_{}_ssim_{}.jpg'.format(idx,epoch, psnr_img, ssim_img)), pred)
    
    psnr_avg = sum(psnr_ls) / len(psnr_ls)
    ssim_avg = sum(ssim_ls) / len(ssim_ls)
    print(f"Valid Epoch: {epoch}, loss_total: {loss_per_epoch}, l1_loss: {l1loss_per_epoch}, msssim: {msssim_per_epoch}, PSNR: {psnr_avg}, SSIM:{ssim_avg}")

    f = open(metric_save_path, 'w' if epoch == 0 else 'a')
    f.write('psnr_avg:{}, ssim_avg:{}, iter:{}\n'.format(psnr_avg,ssim_avg,epoch))
    print('metric average printed.')        
    f.close()

# valid(model, valid_dataloader, 0, img_save_path, csv_save_path)
# --- Training --- #

# """
for epoch in range(cfg.max_iter):

    loss_per_epoch = 0
    l1loss_per_epoch = 0
    msssim_per_epoch = 0
    for idx, data in tqdm(enumerate(train_dataloader), desc = f"Training Epoch: {epoch}", total=round(len(train_dataset)/batch_size)):
        model.train()
        tgt, low = data
        tgt = tgt.to(device)
        low = low.to(device)
        pred = model(low)

        l1_loss = L1_loss(pred, tgt)
        ms_ssim = 1 - MS_SSIM_loss(pred, tgt) #? why 1-msssim

        loss_total = 0.8 * l1_loss + 0.2 * ms_ssim

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        loss_per_epoch += loss_total.item()
        l1loss_per_epoch += l1_loss.item()
        msssim_per_epoch += ms_ssim.item()

    if epoch % val_interval == 0:
        valid(model, valid_dataloader, epoch, img_save_path, csv_save_path)

    if epoch in atWhichSave:
        print("Time to save our model.")
        torch.save({'model': model.state_dict()},os.path.join(weight_save_path,'weights_{}'.format(epoch)))

    if epoch in atWhichReduce:
        for group in optimizer.param_groups:
            old_lr = group['lr']
            group['lr'] = reduce_lr_by*old_lr
            if group['lr']<1e-5:
                group['lr']=1e-5
            print('Changing LR from {} to {}'.format(old_lr,group['lr']))

    print(f"Training Epoch: {epoch}, loss_total = {loss_per_epoch}, l1loss = {l1loss_per_epoch}, msssim = {msssim_per_epoch}")
    train_save_path = "./checkpoint/train_loss.txt"
    f = open(train_save_path, 'w' if epoch == 0 else 'a')
    f.write('loss_total:{}, l1_loss:{}, msssim:{}, epoch:{}\n'.format(loss_per_epoch,l1loss_per_epoch,msssim_per_epoch,epoch))
    # print('metric average printed.')        
    f.close()
# """

