import yaml
from addict import Dict


def read_yaml(fpath):
    with open(fpath, mode="r", encoding="utf-8") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)
