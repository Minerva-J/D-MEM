##########configuration of GPU 
# import os,cv2
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.1
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))
################

# import time
from pathlib import Path
import numpy as np
# import matplotlib.pyplot as plt
import glob
import torch
import torch.nn as nn
# import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from models import tiramisu
from datasets import camvid
# from datasets import joint_transforms
# import utils.imgs
import utils.training as train_utils

CAMVID_PATH = Path('./data')
FILE_test_imgs_original = './data/test'

batch_size = 1
# normalize = transforms.Normalize(mean=camvid.testmean, std=camvid.teststd)
normalize = transforms.Normalize(mean=camvid.testmean, std=camvid.teststd)
imgNumber = len(glob.glob(FILE_test_imgs_original + "/*.png"))
names = []
for files in glob.glob(FILE_test_imgs_original + "/*.png"):
    # print('name:',files[12:])
    names.append(files[12:])
test_dset = camvid.CamVid(
    CAMVID_PATH, 'test', joint_transform=None,
    transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dset, batch_size=batch_size, shuffle=False)

print("Test: %d" %len(test_loader.dataset.imgs))

inputs, targets = next(iter(test_loader))
print("Inputs: ", inputs.size())
print("Targets: ", targets.size())

# utils.imgs.view_image(inputs[0])
# utils.imgs.view_annotated(targets[0])

##############train
torch.cuda.manual_seed(0)
# model = tiramisu.FCDenseNet67(n_classes=4).cuda()
# criterion = nn.NLLLoss2d(weight=camvid.class_weight.cuda()).cuda()
###########test		
model=torch.load("/home/jiezhao/Desktop/1/1/Pap-Smear-Nucleus-Segmentation/repo/pytorch_tiramisu-master/weights/Fc-Densenet67/1361_0.36358_0.87145_model.pkl")	
	
since = time.time()				 
# train_utils.test(model, test_loader, criterion, epoch=1)
# train_utils.view_sample_predictions(model, test_loader, n=1)
train_utils.view_sample_predictions(model, test_loader, names)
time_elapsed = time.time() - since
print('Train Time {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))