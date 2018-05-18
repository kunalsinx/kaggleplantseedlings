import math
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from configs.plantseed_config import config


# def imshow(img):
# 	opt = config()
# 	mean = list(opt.mean)
# 	std = list(opt.std)
# 	for i in range(3):
# 		img[i] = img[i]*std[i] + mean[i]
# 	npimg = img.numpy()
# 	plt.figure(figsize=(14,14))
# 	plt.imshow(np.transpose(npimg, (1, 2, 0)))

def imshow(images, labels, classes, imgs_per_row=4):
    """
    images - stack of images 
    pred_prob - contains list of probabilities and predicted class(incase of incorrect predictions)
    """
    opt = config()
    mean = list(opt.mean)
    std = list(opt.std)
    for img in images:
    	for i in range(3):
    		img[i] = img[i]*std[i] + mean[i]
    pil_convertor = torchvision.transforms.ToPILImage(mode='RGB')
    pil_images = [ pil_convertor(img) for img in images ]
    batches = math.ceil(len(pil_images)/float(imgs_per_row))
    for i in range(batches):
        imgs = pil_images[i*imgs_per_row:(i+1)*imgs_per_row]
        lab = labels[i*imgs_per_row:(i+1)*imgs_per_row]
        fig, ax = plt.subplots(nrows=1, ncols=len(imgs), sharex="col", sharey="row", figsize=(4*(len(imgs)),4), squeeze=False)
        for i, img in enumerate(imgs):    
            ax[0,i].imshow(img)
            ax[0,i].set_title(classes[lab[i].item()])

def vis_unnormalise(dataloaders, classes):
	# dataloaders, dataset_sizes, classes = plantdataset.getdataloader("input/")
	dataiter = iter(dataloaders['train'])
	images, labels = dataiter.next()

	imshow(images[:4], labels[:4], classes)
	#print(' '.join('%5s \t\t' %labels[j] for j in range(4)))