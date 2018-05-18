import os
import torch
import math
import torchvision

from PIL import Image
from operator import itemgetter
from configs.plantseed_config import config
from matplotlib import pyplot as plt
from functions.dataloaders import ListDataset

def imshow(images, pred_prob, imgs_per_row=3):
    """
    images - stack of images 
    pred_prob - contains list of probabilities and predicted class(incase of incorrect predictions)
    """
    pil_convertor = torchvision.transforms.ToPILImage(mode='RGB')
    pil_images = [ pil_convertor(img) for img in images ]
    batches = math.ceil(len(pil_images)/float(imgs_per_row))
    for i in range(batches):
        imgs = pil_images[i*imgs_per_row:(i+1)*imgs_per_row]
        lab = pred_prob[i*imgs_per_row:(i+1)*imgs_per_row]
        fig, ax = plt.subplots(nrows=1, ncols=len(imgs), sharex="col", sharey="row", figsize=(4*(len(imgs)),4), squeeze=False)
        for i, img in enumerate(imgs):    
            ax[0,i].imshow(img)
            ax[0,i].set_title(lab[i])

class error_checking():
    def __init__(self):
        opt = config()
        train_dir = opt.home_loc + "input/train"
        classes = os.listdir(train_dir)
        classes = sorted(classes, key = lambda item: (int(item.partition(' ')[0]) if item[0].isdigit() else float('inf'), item))
        best_pred = {}
        worst_pred = {}
        for i in classes:
            best_pred[i] = []
            worst_pred[i] = []
        
        opt = config()
        opt.normalize = False
        valset = ListDataset(opt, train = "valid")
        transforms = valset.test_transformations(opt)
        correct_pred = open(opt.inference_correct_loc, 'r')
        incorrect_pred = open(opt.inference_incorrect_loc, 'r')

        for row in incorrect_pred:
            list_ = row.strip(" \n").split(", ")
            image = Image.open(list_[0]).convert('RGB')
            image = transforms(image)
            worst_pred[list_[1]].append((image, list_[2], float(list_[3])))
        
        for row in correct_pred:
            list_ = row.strip(" \n").split(", ")
            image = Image.open(list_[0]).convert('RGB')
            image = transforms(image)
            best_pred[list_[1]].append((image, float(list_[3])))
        
        self.best_pred = best_pred
        self.worst_pred = worst_pred
        
    def worst_prediction(self, label, num=3, imgs_per_row=4):
        '''
        Given a label aka class displays the top incorrect predictions for that class
        '''
        worst_list = self.worst_pred[label]
        if(len(worst_list)<num):
            print('Requested {} but total number of incorrect predictions is {}'.format(num, len(worst_list)))
            num = len(worst_list)
        print("Top {} worst predictions for {}:".format(num,label))
        worst_list = sorted(worst_list, key=itemgetter(2), reverse=True )
        pred_prob = [(c,round(p,3)) for _,c,p in worst_list[:num]]
        images = torch.stack([image.cpu() for image,_,m_ in worst_list[:num]])
        imshow(images, pred_prob, imgs_per_row = imgs_per_row)
        
    def best_prediction(self, label, num=3, imgs_per_row=4):
        '''
        Given a label aka class displays the top correct predictions for that class
        '''
        best_list = self.best_pred[label]
        if(len(best_list)<num):
            print('Requested {} but total number of correct predictions is {}'.format(num, len(best_list)))
            num = len(best_list)

        print("Top {} best predictions for {}:".format(num,label))
        best_list = sorted(best_list, key=itemgetter(1), reverse=True )
        pred_prob = [round(p,3) for _,p in best_list[:num]]
        images = torch.stack([image.cpu() for image,_ in best_list[:num]])
        imshow(images, pred_prob, imgs_per_row = imgs_per_row)