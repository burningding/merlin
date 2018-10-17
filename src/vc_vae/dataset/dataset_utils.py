import numpy as np


def make_one_hot(labels, n_class):
    assert(np.asarray(labels).ndim == 1)
    one_hot = np.zeros((len(labels), n_class), dtype=np.float32)
    for i, label in enumerate(labels):
        one_hot[i, label] = 1.
    return one_hot

def read_binfile(filename, dim=60, dtype=np.float32):
    '''
    Reads binary file into numpy array.
    '''
    fid = open(filename, 'rb')
    v_data = np.fromfile(fid, dtype=dtype)
    fid.close()
    if np.mod(v_data.size, dim) != 0:
        raise ValueError('Dimension provided not compatible with file size.')
    m_data = v_data.reshape((-1, dim)).astype('float32')  # This is to keep compatibility with numpy default dtype.
    m_data = np.squeeze(m_data)
    return m_data

def write_binfile(m_data, filename, dtype=np.float32):
    '''
    Writes numpy array into binary file.
    '''
    m_data = np.array(m_data, dtype)
    fid = open(filename, 'wb')
    m_data.tofile(fid)
    fid.close()
    return