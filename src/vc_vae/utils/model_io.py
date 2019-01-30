import torch
import os
import pickle


def save_checkpoint(state, folder='./exp/arctic/model', filename='model.pth'):
    """
    Save the models in training process
    :param state: state dictionary
    :param filename: saved filename
    :return: None
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, filename)
    torch.save(state, filename)


def save_pitch_model(model, filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # filename = os.path.join(folder, filename)
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def load_pitch_model(filename):
    if not os.path.exists(filename):
        raise RuntimeError('pitch model does not exist, please run train_val.py to build the pitch model first')
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model