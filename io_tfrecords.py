import json

import imageio
import numpy as np
import tensorflow as tf

def video2tfrecords(video_path, json_path, classes):
    video = imageio.get_reader(video_path,  'ffmpeg')
    frames = np.array([frame for frame in video], dtype=np.uint8)
    frames_bytes = tf.io.serialize_tensor(frames).numpy()
    frames_shape = tf.convert_to_tensor(frames.shape, dtype=tf.int64)
    with open(json_path, 'r') as f:
        json_dict = json.load(f)
    count_list = []
    for label in classes:
        count_list.append(json_dict[label] if label in json_dict.keys() else 0)
    feature = {
        'video': tf.train.Feature(bytes_list=tf.train.BytesList(value=[frames_bytes])),
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=frames_shape.numpy())),
        'count':  tf.train.Feature(int64_list=tf.train.Int64List(value=count_list)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))