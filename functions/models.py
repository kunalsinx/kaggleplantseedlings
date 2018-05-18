import torch
import torchvision


def resnet50(num_classes, pretrained=True):
	model = torchvision.models.resnet50(pretrained=pretrained)
	num_fltrs = model.fc.in_features
	model.fc = torch.nn.Linear(num_fltrs, num_classes)
	return model

def resnet18(num_classes, pretrained=True):
	model = torchvision.models.resnet18(pretrained=pretrained)
	num_fltrs = model.fc.in_features
	model.fc = torch.nn.Linear(num_fltrs, num_classes)
	return model