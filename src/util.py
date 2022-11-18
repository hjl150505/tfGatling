import tensorflow as tf
import io


def get_file_buffer_by_tfds(rp):
    data = tf.data.TextLineDataset([rp])
    data = [x.decode() for x in data.as_numpy_iterator()]
    data = '\n'.join(data)
    data = io.StringIO(data)

    return data
