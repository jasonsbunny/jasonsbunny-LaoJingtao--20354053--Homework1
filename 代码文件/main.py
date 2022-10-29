# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 16:50:15 2022

@author: nkliu
"""

import torch
import model_SSD
import load_data
import train

batch_size = 32


if __name__ =='__main__':
    #读取训练集数据
    train_iter = load_data.load_data(batch_size)
    net = model_SSD.TinySSD(num_classes=1)
    net = net.to('cpu')

    #训练模型
    train.main(net,train_iter)