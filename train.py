# -*- coding: utf-8 -*-
##########configuration of GPU 
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
################

import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import tiramisu
from datasets import camvid
from datasets import joint_transforms
import utils.imgs
import utils.training as train_utils

# CAMVID_PATH = Path('/bigguy/data', 'SegNet-Tutorial/CamVid')
CAMVID_PATH = Path('./data')
RESULTS_PATH = Path('results/')
WEIGHTS_PATH = Path('weights/')
RESULTS_PATH.mkdir(exist_ok=True)
WEIGHTS_PATH.mkdir(exist_ok=True)
batch_size = 1

normalize = transforms.Normalize(mean=camvid.mean, std=camvid.std)
train_joint_transformer = transforms.Compose([
    #joint_transforms.JointRandomCrop(224), # commented for fine-tuning
    joint_transforms.JointRandomHorizontalFlip()
    ])
train_dset = camvid.CamVid(CAMVID_PATH, 'train',##########like /home/jiezhao/1/1.jpg
      joint_transform=train_joint_transformer,
      transform=transforms.Compose([
          transforms.ToTensor(),
          normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dset, batch_size=batch_size, shuffle=True)

val_dset = camvid.CamVid(
    CAMVID_PATH, 'val', joint_transform=None,
    transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]))
val_loader = torch.utils.data.DataLoader(
    val_dset, batch_size=batch_size, shuffle=False)

test_dset = camvid.CamVid(
    CAMVID_PATH, 'test', joint_transform=None,
    transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dset, batch_size=batch_size, shuffle=False)
	
print("Train: %d" %len(train_loader.dataset.imgs))
print("Val: %d" %len(val_loader.dataset.imgs))
# print("Test: %d" %len(test_loader.dataset.imgs))
print("Classes: %d" % len(train_loader.dataset.classes))

inputs, targets = next(iter(train_loader))
print("Inputs: ", inputs.size())
print("Targets: ", targets.size())

# utils.imgs.view_image(inputs[0])
# utils.imgs.view_annotated(targets[0])


##############train
LR = 1e-4
LR_DECAY = 0.995
DECAY_EVERY_N_EPOCHS = 1
N_EPOCHS = 10000
torch.cuda.manual_seed(0)
def init_conv_offset(m):
    m.weight.data = torch.zeros_like(m.weight.data)
    if m.bias is not None:
        m.bias.data = torch.FloatTensor(m.bias.shape[0]).zero_()

# model = tiramisu.FCDenseNet_test72(n_classes=4).cuda()
model = tiramisu.FCDenseNet_36(n_classes=4).cuda()
model.apply(train_utils.weights_init)
# model.offsets.apply(init_conv_offset)
# print(model)
# model = nn.DataParallel(model,device_ids=[0, 1])
# model.summary()
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.NLLLoss(weight=camvid.class_weight.cuda()).cuda()
net = model
params = list(net.parameters())  
k = 0  
for i in params:  
	l = 1  
	# print("该层的结构：" + str(list(i.size())))  
	for j in i.size():  
		l *= j  
	# print("该层参数和：" + str(l))  
	k = k + l  
print("总参数数量和：" + str(k))  
best_loss = 2
best_acc = 0.7
for epoch in range(1, N_EPOCHS+1):
    since = time.time()

    ### Train ###
    trn_loss, trn_err = train_utils.train(
        model, train_loader, optimizer, criterion, epoch)
    print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}'.format(
        epoch, trn_loss, 1-trn_err))    
    time_elapsed = time.time() - since  
    print('Train Time {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    ### Test ###
    val_loss, val_err = train_utils.test(model, val_loader, criterion, epoch)    
    print('Val - Loss: {:.4f} | Acc: {:.4f}'.format(val_loss, 1-val_err))
    time_elapsed = time.time() - since  
    print('Total Time {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    ### Checkpoint ###    
    val_acc = 1-val_err
    print(val_acc,best_acc)
    if val_acc > best_acc:
	    print('weight saved,val_acc:',val_acc)   
	    # train_utils.save_weights(model, epoch, val_loss, 1-val_err)
	    best_acc = val_acc

    ### Adjust Lr ###
    train_utils.adjust_learning_rate(LR, LR_DECAY, optimizer, 
                                     epoch, DECAY_EVERY_N_EPOCHS)

    ###########test								 
    train_utils.test(model, test_loader, criterion, epoch=1)
    train_utils.view_sample_predictions(model, test_loader, n=1)


