##GLNet: generate high-light image
import torch
from model.unet import UNet as UNet
from model.drbn import DRBN as DRBN
def make_model(model_name):
    if model_name == "UNet":
        return UNet(3, 3)
    if model_name == "DRBN":
        model = DRBN()
        model.load_state_dict(torch.load('pre_model/model_s1.pt'))
        return model 

