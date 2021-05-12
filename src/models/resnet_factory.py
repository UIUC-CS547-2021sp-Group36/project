"""
Only put here to avoid a circular dependency
"""
import torchvision.models as tvmodels

def create_resnet(resnet, pretrained=True):
    if resnet == "resnet18":
        return tvmodels.resnet18(pretrained=pretrained)
    elif resnet == "resnet18":
        return tvmodels.resnet18(pretrained=pretrained)
    elif resnet == "resnet50":
        return tvmodels.resnet50(pretrained=pretrained)
    elif resnet == "resnet101":
        return tvmodels.resnet101(pretrained=pretrained)
    elif resnet == "resnet152":
        return tvmodels.resnet152(pretrained=pretrained)
    else:
        raise NotImplemented("I'm sorry, couldn't create inner model {}".format(resnet_name))

def find_resnet_out_features(resnet):
    n_resnetout = list(resnet.children())[-1].out_features
    return n_renetout
