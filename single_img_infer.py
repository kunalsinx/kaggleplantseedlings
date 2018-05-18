import os
import sys
import torch

from PIL import Image
from functions import models
from functions import train_model
from functions import dataloaders
from configs.plantseed_config import config
from functions.dataloaders import ListDataset

'''
Used for doing inference on single page. The location of the image should be given as arguments
'''

def main(argv):

	opt = config()
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


	with torch.no_grad():
	    fullname = argv[1]
	    image = Image.open(fullname).convert('RGB')
	    image = valtransforms(image)
	    image = image.unsqueeze(0).to(device)
	    #print(image.size())
	    output = model_conv(image)
	    output = torch.nn.functional.softmax(output)
	    probs, predicted = torch.max(output, 1)

	print("The given image is {} with probability {:3f}".format(classes[predicted.item()], probs.item()))




if __name__ == '__main__':
	main(sys.argv)