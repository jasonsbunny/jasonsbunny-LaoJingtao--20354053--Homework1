# Target Detection——TinySSD
## 1. Environment Configuration


在本次目标检测任务中，我使用Anaconda来配置深度学习的环境，因为Anaconda中集成了大部分我们在编写代码时会经常使用的第三方库，并且可以很方便地获取需要的其他库并且进行管理。同时，也很方便对现存的库进行管理。该目标检测任务基于PyTorch框架来搭建深度神经网络模型，我所下载的PyTorch版本是Windows-Conda-Python-Cuda10.2，对应的Python版本是3.8，使用的开发工具是PyCharm。  

（1）基于PyTorch深度学习框架，任务使用其中的torch.nn模块来搭建网络并进行模型训练，例如torch.nn.functional包含大量的损失函数和激活函数可供使用。  

（2）torchvision是独立于pytorch的关于图像操作的一些方便工具库，任务使用torchvision库中的函数读取训练集数据以及对图像进行常见的图像处理操作。  

===================================================  

## 2. Traing Process

（1）生成多尺度的先验框
将图片输入到网络，生成多尺度的先验框，并为每个锚框预测类别和偏移量。该过程通过anchors, cls_preds, bbox_preds = net(X)完成。  


（2）先验框匹配
在训练过程中，首先要确定训练图片中的ground truth（真实目标）与哪个先验框来进行匹配，与之匹配的先验框所对应的边界框将负责预测它。SSD的先验框与ground truth的匹配原则主要有两点。首先，对于图片中每个ground truth，找到与其IOU最大的先验框，该先验框与其匹配，这样，可以保证每个ground truth一定与某个先验框匹配。对于剩余的未匹配先验框，若某个ground truth的 IOU大于某个阈值（一般是0.5），那么该先验框也与这个ground truth进行匹配。
为了保证正负样本尽量平衡，SSD采用了hard negative mining，就是对负样本进行抽样，抽样时按照置信度误差（预测背景的置信度越小，误差越大）进行降序排列，选取误差的较大的top-k作为训练的负样本，以保证正负样本比例接近1:3。
该过程通过bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)完成。  

（3）损失函数
根据类别和偏移量的预测和标注值计算损失函数，然后利用损失函数值更新网络中的参数值。该过程通过l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,bbox_masks)， l.mean().backward()完成。对于一定的epoch，保存其模型参数。  

===================================================  

## 3. Test Method  
打开test.py文件，将数据文件路径更改为自己的数据文件路径运行即可。
