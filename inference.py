import os
import torch

from PIL import Image
from functions import models
from functions import train_model
from functions import dataloaders
from configs.plantseed_config import config
from functions.dataloaders import ListDataset

'''
given a validation txt file containing locations of images and labels, outputs two files - correct_prediction.txt and 
																						   incorrect_prediction.txt 
'''
opt = config()
#print("Load dataloaders....")
valtransforms= ListDataset.test_transformations(opt)

train_dir = opt.home_loc + "input/train"
classes = os.listdir(train_dir)
classes = sorted(classes, key = lambda item: (int(item.partition(' ')[0]) if item[0].isdigit() else float('inf'), item))

model_conv = models.resnet50(12, pretrained = True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_conv = model_conv.to(device)

checkpoint = torch.load(opt.save_loc + 'checkpoint.pth.tar')
model_conv.load_state_dict(checkpoint['state_dict'])
model_conv = model_conv.eval()

valid = open(opt.val_data_loc, "r")
correct_preds = open(opt.inference_correct_loc, "w+")
incorrect_preds = open(opt.inference_incorrect_loc, "w+")

with torch.no_grad():
    for row in valid:
        fullname = row.strip(" \n").split(", ")[0]
        label = int(row.strip(" \n").split(", ")[1])
        image = Image.open(fullname).convert('RGB')
        image = valtransforms(image)
        image = image.unsqueeze(0).to(device)
        #print(image.size())
        output = model_conv(image)
        output = torch.nn.functional.softmax(output)
        probs, predicted = torch.max(output, 1)
        #print(predicted.item(), label)
        #print(probs, predicted)
        if predicted.item()==label:
            correct_preds.write("{}, {}, {}, {} \n".format(fullname, classes[label], classes[label], probs.item()))
        else:
            incorrect_preds.write("{}, {}, {}, {} \n".format(fullname, classes[label], classes[predicted.item()], probs.item()))
correct_preds.close()
incorrect_preds.close()
