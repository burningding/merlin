from torch.utils.data import DataLoader
import torch
import re
from torch._six import string_classes, int_classes
import collections

def _frame_collate_fn(batch):
    """
    merge multiple sequences into one long sequence
    :param batch: sequences[#sequence in a batch][#frames][#dim]
    :return: merged sequence[#sequence in batch x #frames][#dim]
             merged label[#sequence in batch x #frames][1]
    """

    sequences, labels = zip(*batch)
    _frames = [len(seq) for seq in sequences]
    frames = sum(_frames)
    fea_dim = sequences[0].shape[1]
    lab_dim = labels[0].shape[1]
    cat_seqs = torch.zeros(frames, fea_dim).float()
    cat_labs = torch.zeros(frames, lab_dim).float()
    end = 0
    for i, seq in enumerate(sequences):
        length = _frames[i]
        cat_seqs[end: end + length, :] = seq
        label = labels[i]
        cat_labs[end: end + length, :] = label.repeat(length, 1)
        end += length

    permute = torch.randint(high=frames, size=(len(batch) * 500,)).long()
    cat_seqs = cat_seqs[permute, :]
    cat_labs = cat_labs[permute, :]
    return cat_seqs, cat_labs


class FrameDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets in frames.
        """
        super(FrameDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _frame_collate_fn
