import os
import glob
import random 

seed_ = 7
random.seed(seed_)
'''
Used for generating train, validation and test txt files containing the locations of the images and the labels accordingly.
File struct should be - input/train/class/images.png
										  img2.png
'''

def gen_train_valid(data_dir, split =0.1):
	train_dir = data_dir + "train/"
	classes = os.listdir(train_dir)
	classes = sorted(classes, key = lambda item: (int(item.partition(' ')[0]) if item[0].isdigit() else float('inf'), item))
	x_train = open("data/x_train.txt", "w+")
	x_valid = open("data/x_valid.txt", "w+")

	for i,label in enumerate(classes):
	    files = glob.glob(data_dir)
	    files = glob.glob(data_dir+"train/"+label+"/*")
	    random.shuffle(files)
	    train = files[:int((1-split)*len(files))]
	    valid = files[int((1-split)*len(files)):]
	    for file in train:
	        x_train.write("{}, {} \n".format(file, i))
	    for file in valid:
	        x_valid.write("{}, {} \n".format(file, i))
	        
	x_train.close()
	x_valid.close()

def gen_test(data_dir):
	x_test = open("data/x_test.txt", "w+")
	test_dir = data_dir + "test"
	files = glob.glob(test_dir+"/*")
	for file in files:
		x_test.write("{} \n".format(file))
	x_test.close()