import time
import torch
import shutil

from configs.plantseed_config import config


def train(param):
	'''
	Used for training the model
	'''
	model = param["model"] 
	criterion = param["criterion"]
	optimizer = param["optimizer"]
	scheduler = param["scheduler"]
	dataloaders = param["dataloaders"]
	dataset_sizes = param["dataset_sizes"]
	use_checkpoint = param["use_checkpoint"]
	num_epochs = param["epoch"]
	device = param["device"]

	since = time.time()
	best_model_wts = model.state_dict()
	best_acc = 0.0
	train_loss = []
	train_acc = []
	val_loss = []
	val_acc = []
	get_lr = []
	opt = config()

	if use_checkpoint:
		checkpoint = torch.load(opt.save_loc+'checkpoint.pth.tar')
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		scheduler.load_state_dict(checkpoint['scheduler'])
		prev = checkpoint['prev']
		# start_epochs = checkpoint['epoch']
		# num_epochs = num_epochs-start_epochs
		# print("setting num_epochs to {}".format(num_epochs))
		print("Previous best accuracy {}".format(prev))

	for epoch in range(num_epochs):
	    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
	    print('-' * 10)

	    # Each epoch has a training and validation phase
	    for phase in ['train', 'valid']:
	        if phase == 'train':
	            scheduler.step()
	            get_lr.append(scheduler.get_lr())
	            model.train()  # Set model to training mode
	        else:
	            model.eval()   # Set model to evaluate mode

	        running_loss = 0.0
	        running_corrects = 0

	        # Iterate over data.
	        for inputs, labels in dataloaders[phase]:
	            inputs = inputs.to(device)
	            labels = labels.to(device)
	            labels = labels.squeeze(1)

	            # zero the parameter gradients
	            optimizer.zero_grad()

	            # forward
	            # track history if only in train
	            with torch.set_grad_enabled(phase == 'train'):
	                outputs = model(inputs)
	                _, preds = torch.max(outputs, 1)
	                loss = criterion(outputs, labels)

	                # backward + optimize only if in training phase
	                if phase == 'train':
	                    loss.backward()
	                    optimizer.step()

	            # statistics
	            running_loss += loss.item() * inputs.size(0)
	            running_corrects += torch.sum(preds == labels.data)

	        epoch_loss = running_loss / dataset_sizes[phase]
	        epoch_acc = running_corrects.double() / dataset_sizes[phase]
	        
	        if phase == "train":
	            train_loss.append(epoch_loss)
	            train_acc.append(epoch_acc.item())
	        else:
	            val_loss.append(epoch_loss)
	            val_acc.append(epoch_acc.item())
	        

	        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
	            phase, epoch_loss, epoch_acc))
	        # copy the mode
	#             print(epoch_acc.item(), best_acc)
	        if phase == 'valid' and epoch_acc.item() > best_acc :
	            best_acc = epoch_acc.item()
	            best_model_wts = model.state_dict()
	            save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 
	            				 'optimizer' : optimizer.state_dict(), 'scheduler' : scheduler.state_dict(), 'prev' : best_acc}, opt.save_loc)

	    #print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
	    time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model, train_loss, val_loss, train_acc, val_acc, get_lr

def validate(dataloaders, model, criterion, device):
	'''
	Used for validation after training is over and the model should have the trained weights
	'''
	val_loader = dataloaders['valid']
	running_loss = 0.0
	running_corrects = 0

	for inputs, labels in val_loader:
		inputs = inputs.to(device)
		labels = labels.to(device)
		labels = labels.squeeze(1)

		with torch.no_grad():
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			loss = criterion(outputs, labels)
		
		running_loss += loss.item() * inputs.size(0)
		running_corrects += torch.sum(preds == labels.data)
		# print(running_corrects.double())
	size = len(val_loader.dataset)
	loss = running_loss / size
	acc = running_corrects.double() / size
	print('Val Acc: {:4f}'.format(acc))

def classwise_accuracy(model_conv, dataloaders, classes, device):
	'''
	prints classwise accuracy and again the model should have the trained weights
	'''
	class_correct = list(0. for i in range(len(classes)))
	class_total = list(0. for i in range(len(classes)))
	with torch.no_grad():
	    for images, labels in dataloaders['valid']:
	        images, labels = images.to(device), labels.to(device)
	        labels = labels.squeeze(1)
	        outputs = model_conv(images)
	        _, predicted = torch.max(outputs, 1)
	        c = (predicted == labels).squeeze()
	        for i in range(len(labels)):
	            label = labels[i]
	            class_correct[label] += c[i].item()
	            class_total[label] += 1


	for i in range(12):
	    print('Accuracy of %5s : %2d %%' % (
	        classes[i], 100 * class_correct[i] / class_total[i]))


def save_checkpoint(state, loc, filename='checkpoint.pth.tar'):
    filename = loc+filename
    torch.save(state, filename)
    modelfile = loc + 'model_best.pth.tar'
    shutil.copyfile(filename, modelfile)