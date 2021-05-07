#This is just initial experiment code for now.
import torchvision as tv
import torch
import torchvision.models as models
import torchvision.models as tvmodels

from .ResnetFrozenWrapper import ResnetFrozenWrapper
from .NewModel import NewModel

def create_model(model_name="dummy"):
    if "resnet18" == model_name:
        return tvmodels.resnet18(pretrained=True)
    elif model_name in ["newModel","NewModel"]:
        return NewModel()
    elif "LowDNewModel" == model_name:
        return NewModel(out_features=30)
    elif model_name in ["dummy", "resnet18"]:
        return ResnetFrozenWrapper(resnet="resnet18")
    elif model_name in ["dummy30", "resnet18_30"]:
        return ResnetFrozenWrapper(resnet="resnet18",out_features=30,internal_dimension=150)
    elif "resnet101" == model_name:
        return ResnetFrozenWrapper(resnet="resnet101")
    elif "resnet101_30" == model_name:
        return ResnetFrozenWrapper(resnet="resnet101",out_features=30,internal_dimension=150)

    #TODO: Add options for other models as we implement them.

    raise Exception("No Model Specified")
