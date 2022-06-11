import os
import numpy as np

class configs(object):
    def __init__(self, base_lr, reduce_lr_by, atWhichReduce, batch_size, atWhichSave, max_iter, debug, debug_iter, debug_size,
                metric_avg_file, test_amp_file, train_amp_file,
                weight_save_path, img_save_path, csv_save_path, sony_img_path, 
                ):
        # --- Model --- #
        self.base_lr = base_lr # base learning rate
        self.reduce_lr_by = reduce_lr_by # learning rate reduce %
        self.atWhichReduce = atWhichReduce # at which epoch will lr reduce
        self.batch_size = batch_size  
        self.atWhichSave = atWhichSave # at which epoch will model be saved
        self.max_iter = max_iter 

        # --- Debug --- #
        self.debug = debug # debug mode
        self.debug_iter = debug_iter # max iter in debug mode
        self.debug_size = debug_size

        if self.debug == True:
            self.max_iter = self.debug_iter # using debug iter

        # --- Save paths --- #
        self.metric_avg_file = metric_avg_file
        self.test_amp_file = test_amp_file
        self.train_amp_file = train_amp_file
        self.weight_save_path = weight_save_path
        self.img_save_path = img_save_path
        self.csv_save_path = csv_save_path

        self.sony_img_path = sony_img_path
        self.train_list_path = os.path.join(self.sony_img_path, "Sony_train_list.txt")
        self.valid_list_path = os.path.join(self.sony_img_path, "Sony_val_list.txt")
        self.test_list_path = os.path.join(self.sony_img_path, "Sony_test_list.txt")

        

cfg = configs(
    base_lr = 1e-4, 
    reduce_lr_by = 0.1, 
    atWhichReduce = [500000], 
    batch_size = 8, 
    atWhichSave = [2,100002,150002,200002,250002,300002,350002,400002,450002,500002,550000, 600000,650002,700002,750000,800000,850002,900002,950000,1000000], 
    max_iter = 1000005, 
    debug = True, 
    debug_iter = 10000005,
    debug_size = 30,
    metric_avg_file = "./checkpoint/metric_average.txt", 
    test_amp_file = "./checkpoint/test_amp.txt", 
    train_amp_file = "./checkpoint/train_amp.txt",
    weight_save_path = "./checkpoint/weights", 
    img_save_path = "./checkpoint/images", 
    csv_save_path = "./checkpoint/csv_files.txt",
    sony_img_path = "F:/Learning-to-see-in-the-dark/Sony" 
)
