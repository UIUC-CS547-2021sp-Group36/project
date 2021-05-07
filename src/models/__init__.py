#This is just initial experiment code for now.
import torchvision as tv
import torch
import torchvision.models as models
import torchvision.models as tvmodels

from .Resnet18FrozenWrapper import Resnet18FrozenWrapper
from .Resnet101FrozenWrapper import Resnet101FrozenWrapper
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
    elif "dummy30" == model_name:
        return Resnet18FrozenWrapper(output_dimension=30,internal_dimension=150)
    elif "resnet101" == model_name:
        return Resnet101FrozenWrapper()
    elif "resnet101_30" == model_name:
        return Resnet101FrozenWrapper(output_dimension=30,internal_dimension=150)

    #TODO: Add options for other models as we implement them.

    raise Exception("No Model Specified")
