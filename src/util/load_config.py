import yaml

def load(fn):
    with open(fn,"r") as stream:
        return yaml.load(stream,Loader=yaml.Loader)
