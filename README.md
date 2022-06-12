# Implementation-of-Restoring-Extremely-Dark-Images-in-Real-Time
Unofficial Implementation of Restoring Extremely Dark Images in Real Time 

Computational Photography Course Project

Paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Lamba_Restoring_Extremely_Dark_Images_in_Real_Time_CVPR_2021_paper.pdf

Official Implementation: https://github.com/MohitLamba94/Restoring-Extremely-Dark-Images-In-Real-Time

SID dataset https://github.com/cchen156/Learning-to-See-in-the-Dark

## Usage

### Configuration

Go to `./config.py`

| Variables          | Default                              | Meaning                                                      |
| ------------------ | ------------------------------------ | ------------------------------------------------------------ |
| `base_lr`          | `1e-4`                               | Base learning rate for the model                             |
| `reduce_lr_by`     | `0.1`                                | If [atWhichReduce] epoch comes, lr will reduce by [reduce_lr_by] |
| `atWhichReduce`    | `[500000]`                           | At which epoch will lr reduce                                |
| `batch_size`       | `8`                                  | Batch size                                                   |
| `atWhichSave`      | `[2,100002,150002,200002,250002,...` | At which epoch will model be saved                           |
| `max_iter`         | `1000005`                            | Max training iterations                                      |
| `debug`            | `True`                               | Debug Mode activated or not                                  |
| `debug_iter`       | `100`                                | If in debug mode, how many training iterations will be operated |
| `debug_size`       | `150`                                | If in debug mode, how many training images and valid images will be loaded # NOTICE: total loaded img num = 2*debug_size |
| `metric_avg_file`  | `"./checkpoint/metric_average.txt"`  | No use for now                                               |
| `test_amp_file`    | `"./checkpoint/test_amp.txt"`        | No use for now                                               |
| `train_amp_file`   | `"./checkpoint/train_amp.txt"`       | No use for now                                               |
| `weight_save_path` | `"./checkpoint/weights"`             | Where the model will be saved                                |
| `img_save_path`    | `"./checkpoint/images"`              | Where the intermediate processing images will be saved       |
| `csv_save_path`    | `"./checkpoint/csv_files.txt"`       | No use for now                                               |
| `sony_img_path`    | `./Sony`                             | Dataset path                                                 |

==Notice:==

Current code will consume much RAM in preprocessing (Amplifier module in this code `dataloader.py -> get_tgt_and_low()`). I still cannot figure out how to cut this memory usage.

For instance, setting `debug_size` to `150` will consume about 55GB RAM containing 150 images in training set and 150 images in validation set. So if your RAM capacity exceeded, ==please try smaller `debug_size`==.

### Tree

├── Implementation-of-Restoring-Extremely-Dark-Images-in-Real-Time
│   ├── LICENSE
│   ├── README.md
│   ├── __pycache__
│   ├── checkpoint
│   ├── config.py
│   ├── dataloader.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── Sony
│   ├── long
│   └── short
├── Sony_test_list.txt
├── Sony_train_list.txt
├── Sony_val_list.txt

==Notice:== If your dataset is in another directory, please modify `sony_img_path`.

### Training

```
python train.py
```

### 一些疑问

1. 作者在原文中提到 他使用了 ”Intel Xeon E5- 1620V4 CPU with 64GB RAM and GTX 1080Ti GPU to implement our network. “ 但是我自己的测试结果是 64G甚至不能载入全部的数据集，我就很好奇他是咋弄的。
2. 作者还提到了并行优化，但是在代码里完全没有体现，具体可以参看`model.py -> forward()`


## Contribution:
1. New DL arch for low-light single image restoration: faster, cheaper and fewer operations
2. Parallelism strategy with no effect on restoration
3. Modification to the RDB for better restoration
4. New amplifier module which estimates only from the input image without fine-tuning
5. Better generalization capability

## Need to Know:
1. Residual Dense Block (Clicked)

RDB(Residual Dense Block): https://blog.csdn.net/qq_14845119/article/details/81459859

2. Pixel-Shuffle operation
3. Depth-wise Concatenated
4. Grouped convolution
5. MS-SSIM Loss
6. SSIM
7. Ma score
8. NIQE score

## Already Know:

1. Multiply-accumulate: 累计乘加运算
2. Bayer 2x2: 拜尔阵列

R G

G B

3. .ARW: ARW是索尼公司独有的一种文件格式，是RAW文件格式的另一种形式，该格式将CCD或CMOS感光元件生成的12位、14位或22位二进制原始感光数据和摄影环境信息、相机程序调整信息整合在一个文件中。https://baike.baidu.com/item/arw/8814364?fr=aladdin

4. Anti-aliasing: 抗锯齿

5. Amp and weighted amp.

Params:

1. Loss = 0.8 x L1 Loss + 0.2 x MS-SSIM Loss

2. LeakyReLU instead of ReLU: negative slope=0.2

3. LSE(Lower Scale Encoder): 1/2 res.

4. MSE(Medium Scale Encoder): 1/8 res.

5. HSE(Higher Scale Encoder): 1/32 res.

6. ADAM optim: default param. for 1000K iters

First 500K iters: lr=1e-4 and thereupon reduced to 1e-5

7. Init: conv layers with MSRA init.

8. Training Augmentations:

Random Crop: 512 x 512 patches

Horizontal and Vertical fiipping

9. Batchsize=8

10. Conditioned all conv layers with [weight normalization]

11. Testing: no aug, full img res., no weight normalization

12. Amplifier: n=128 in Eq.5


## Dataset

SID dataset https://github.com/cchen156/Learning-to-See-in-the-Dark

2 Camera sensors: 
1. Sony \alpha 7S II Bayer sensor + 4256x2848 res. **More Focused**
2. Fujifilm X-Trans sensor + 6032x4032 res.

**Using SID training and testing split**

## Metrics

1. Number or model params.
2. MAC operations
3. Peak RAM utilization: avg over 100 trials
4. Inference Speed and GMAC consumed: both **GPUs** and **CPUs**
5. Restoration quality
6. PSNR/SSIM
7. Ma and NIQE score

**Using the knowledge of the GT exposure mentioned in the SID dataset for pre-amplification and disable parallelism in our network**