# Implementation-of-Restoring-Extremely-Dark-Images-in-Real-Time
Unofficial Implementation of Restoring Extremely Dark Images in Real Time 

Computational Photography Course Project

Paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Lamba_Restoring_Extremely_Dark_Images_in_Real_Time_CVPR_2021_paper.pdf

Official Implementation: https://github.com/MohitLamba94/Restoring-Extremely-Dark-Images-In-Real-Time

SID dataset https://github.com/cchen156/Learning-to-See-in-the-Dark

Contribution:
1. New DL arch for low-light single image restoration: faster, cheaper and fewer operations
2. Parallelism strategy with no effect on restoration
3. Modification to the RDB for better restoration
4. New amplifier module which estimates only from the input image without fine-tuning
5. Better generalization capability

Need to Know:
1. Residual Dense Block (Clicked)

RDB(Residual Dense Block): https://blog.csdn.net/qq_14845119/article/details/81459859

2. Pixel-Shuffle operation
3. Depth-wise Concatenated
4. Grouped convolution
5. MS-SSIM Loss


Already Know:

Multiply-accumulate: 累计乘加运算

Bayer 2x2: 拜尔阵列
R G
G B

.ARW: ARW是索尼公司独有的一种文件格式，是RAW文件格式的另一种形式，该格式将CCD或CMOS感光元件生成的12位、14位或22位二进制原始感光数据和摄影环境信息、相机程序调整信息整合在一个文件中。https://baike.baidu.com/item/arw/8814364?fr=aladdin

Params:

Loss = 0.8 x L1 Loss + 0.2 x MS-SSIM Loss

LeakyReLU instead of ReLU: negative slope=0.2

