from util.video_stream import generate
from specialized_models.small_net import SmallNet
from util.util import move_to_master
import torch


def edge_statistics():
    frame = generate()
    move_to_master(frame)


def edge_normal():

    while True:

        model_file = ""
        model = SmallNet(3)
        pretrained_model = torch.load(model_file, map_location='cpu')
        model.load_state_dict(pretrained_model)

        model1 = SmallNet(3)
        pretrained_model = torch.load(model_file, map_location='cpu')
        model1.load_state_dict(pretrained_model)

        frame = generate()
        if frame is None:
            break
        output = model(frame)
        if output == 1:
            output = model1(frame)
            if output == 1:
                move_to_master(frame)
            else:
                continue
        else:
            continue


edge_normal()
