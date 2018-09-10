import torch
import os


def save_checkpoint(state, folder='./exp/arctic/model', filename='model.pth'):
    """
    Save the models in training process
    :param state: state dictionary
    :param filename: saved filename
    :return: None
    """
    if not os.path.exists(folder):
        os.mkdir(folder)
    filename = os.path.join(folder, filename)
    torch.save(state, filename)
