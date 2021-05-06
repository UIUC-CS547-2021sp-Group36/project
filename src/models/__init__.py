#This is just initial experiment code for now.
import torchvision as tv
import torch
import torchvision.models as models
import torchvision.models as tvmodels

from .Resnet18FrozenWrapper import Resnet18FrozenWrapper
from .NewModel import NewModel

def create_model(model_name="dummy"):
    if "resnet18" == model_name:
        return tvmodels.resnet18(pretrained=True)
    elif model_name in ["newModel","NewModel"]:
        return NewModel()
    elif "LowDNewModel" == model_name:
        return NewModel(out_features=30)
    elif "dummy" == model_name:
        return Resnet18FrozenWrapper()

    #TODO: Add options for other models as we implement them.

    raise Exception("No Model Specified")
