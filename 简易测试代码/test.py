import torch
import predict
import model_SSD
import cv2
import visualize

##test.py在：代码文件/test.py中

img_name='detection/test/2.jpg'

net = model_SSD.TinySSD(num_classes=1)
net = net.to('cpu')
net.load_state_dict(torch.load('net_30.pkl', map_location=torch.device('cpu')))
X = torch.from_numpy(cv2.imread(img_name)).permute(2, 0, 1).unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()
output = predict.predict(X,net)
visualize.display(img, output.cpu(), threshold=0.5)