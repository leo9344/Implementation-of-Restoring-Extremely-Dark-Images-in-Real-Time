# Implementation-of-Restoring-Extremely-Dark-Images-in-Real-Time
Implementation of Restoring Extremely Dark Images in Real Time  
Paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Lamba_Restoring_Extremely_Dark_Images_in_Real_Time_CVPR_2021_paper.pdf
Official Implementation: https://github.com/MohitLamba94/Restoring-Extremely-Dark-Images-In-Real-Time

RDB(Residual Dense Block): https://blog.csdn.net/qq_14845119/article/details/81459859
SID dataset https://github.com/cchen156/Learning-to-See-in-the-Dark

Contribution:
1. New DL arch for low-light single image restoration: faster, cheaper and fewer operations
2. Parallelism strategy with no effect on restoration
3. Modification to the RDB for better restoration
4. New amplifier module which estimates only from the input image without fine-tuning
5. Better generalization capability

Need to Know:
1. Residual Dense Block (Clicked)
2. Pixel-Shuffle operation
3. Depth-wise Concatenated
4. Grouped convolution

Multiply-accumulate: 累计乘加运算
Bayer 2x2: 拜尔阵列
R G
G B
