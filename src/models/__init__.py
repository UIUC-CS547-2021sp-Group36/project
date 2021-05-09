#This is just initial experiment code for now.
import torchvision as tv
import torch
import torchvision.models as models
import torchvision.models as tvmodels

from .ResnetFrozenWrapper import ResnetFrozenWrapper
from .NewModel import NewModel
from .OneEmbModel import OneEmbModel
from .OneEmbModel2 import OneEmbModel2
from .ThreeEmbModel import ThreeEmbModel

def create_model(model_name="dummy"):
    if "resnet18" == model_name:
        return tvmodels.resnet18(pretrained=True)
    elif model_name in ["newModel","NewModel"]:
        return NewModel()
    elif "paper_resnet18" == model_name:
        return NewModel(resnet="resnet18", out_features=100)
    elif "paper_resnet50" == model_name:
        return NewModel(resnet="resnet50", out_features=100)
    elif "paper_resnet101" == model_name:
        return NewModel(resnet="resnet101", out_features=100)
    elif "paper_resnet152" == model_name:
        return NewModel(resnet="resnet152", out_features=100)
    elif "OneEmbModel_resnet18" == model_name:
        return OneEmbModel(resnet="resnet18", out_features=100)
    elif "OneEmbModel_resnet50" == model_name:
        return OneEmbModel(resnet="resnet50", out_features=100)
    elif "OneEmbModel_resnet101" == model_name:
        return OneEmbModel(resnet="resnet101", out_features=100)
    elif "OneEmbModel_resnet152" == model_name:
        return OneEmbModel(resnet="resnet152", out_features=100)
    elif "OneEmbModel2_resnet18" == model_name:
        return OneEmbModel2(resnet="resnet18", out_features=100)
    elif "OneEmbModel2_resnet50" == model_name:
        return OneEmbModel2(resnet="resnet50", out_features=100)
    elif "OneEmbModel2_resnet101" == model_name:
        return OneEmbModel2(resnet="resnet101", out_features=100)
    elif "OneEmbModel2_resnet152" == model_name:
        return OneEmbModel2(resnet="resnet152", out_features=100)
    elif "ThreeEmbModel_resnet18" == model_name:
        return ThreeEmbModel(resnet="resnet18", out_features=100)
    elif "ThreeEmbModel_resnet50" == model_name:
        return ThreeEmbModel(resnet="resnet50", out_features=100)
    elif "ThreeEmbModel_resnet101" == model_name:
        return ThreeEmbModel(resnet="resnet101", out_features=100)
    elif "ThreeEmbModel_resnet152" == model_name:
        return ThreeEmbModel(resnet="resnet152", out_features=100)
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
