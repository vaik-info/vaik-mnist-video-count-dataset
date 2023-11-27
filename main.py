import argparse
import os
import random
import shutil
import json
from PIL import Image
import imageio
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from digit import Digit

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def write(output_sub_dir_path, sample_num, image_max_size, image_min_size, char_max_size, char_min_size, char_max_num,
          char_min_num, x, y, classes, min_frame_num, max_frame_num, min_char_speed, max_char_speed,
          min_char_size_ratio, max_char_size_ratio, occlusion_ratio, max_delay_frame, fps=10):
    os.makedirs(output_sub_dir_path, exist_ok=True)

    for file_index in tqdm(range(sample_num), desc=f'write at {output_sub_dir_path}'):
        default_canvas = np.zeros(
            (random.randint(image_min_size, image_max_size), random.randint(image_min_size, image_max_size), 3),
            dtype=np.uint8)
        frame_num = random.randint(min_frame_num, max_frame_num)
        count_dict = {}
        # init Digit List
        digit_list = []
        canvas = np.zeros(default_canvas.shape, dtype=default_canvas.dtype)
        for char_index in range(random.randint(char_min_num, char_max_num)):
            while True:
                mnist_index = random.randint(0, y.shape[0] - 1)
                if len(classes) > y[mnist_index]:
                    break
            if classes[y[mnist_index]] not in count_dict.keys():
                count_dict[classes[y[mnist_index]]] = 0
            count_dict[classes[y[mnist_index]]] += 1
            digit_list.append(
                Digit(x[mnist_index], default_canvas.shape, min_char_speed, max_char_speed, char_min_size, char_max_size,
                      min_char_size_ratio, max_char_size_ratio, occlusion_ratio, max_delay_frame))
            canvas = digit_list[-1].paste(canvas)

        # create frames
        image_array_list = [canvas]
        for frame_index in range(frame_num):
            canvas = np.zeros(default_canvas.shape, dtype=default_canvas.dtype)
            for a_digit in digit_list:
                canvas = a_digit.paste(canvas)
            image_array_list.append(canvas)

        file_name = f'{os.path.basename(output_sub_dir_path)}_{file_index:09d}'
        imageio.mimwrite(os.path.join(output_sub_dir_path, f'{file_name}.avi'), image_array_list, fps=fps, codec='rawvideo')
        output_json_path = os.path.join(output_sub_dir_path, f'{file_name}.json')
        with open(output_json_path, 'w') as f:
            json.dump(count_dict, f)


def main(output_dir_path, classes_txt_path, train_sample_num, valid_sample_num, image_max_size, image_min_size,
         char_max_size,
         char_min_size, char_max_num, char_min_num, max_delay_frame, min_frame_num, max_frame_num, min_char_speed, max_char_speed,
         min_char_size_ratio, max_char_size_ratio, occlusion_ratio):
    os.makedirs(output_dir_path, exist_ok=True)

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    classes = []
    with open(classes_txt_path) as f:
        for line in f:
            classes.append(line.strip())

    output_train_dir_path = os.path.join(output_dir_path, 'train')
    write(output_train_dir_path, train_sample_num, image_max_size, image_min_size, char_max_size, char_min_size,
          char_max_num, char_min_num, x_train, y_train, classes, min_frame_num, max_frame_num, min_char_speed,
          max_char_speed, min_char_size_ratio, max_char_size_ratio, occlusion_ratio, max_delay_frame)

    output_valid_dir_path = os.path.join(output_dir_path, 'valid')
    write(output_valid_dir_path, valid_sample_num, image_max_size, image_min_size, char_max_size, char_min_size,
          char_max_num, char_min_num, x_test, y_test, classes, min_frame_num, max_frame_num, min_char_speed,
          max_char_speed, min_char_size_ratio, max_char_size_ratio, occlusion_ratio, max_delay_frame)

    shutil.copy(classes_txt_path, os.path.join(output_dir_path, os.path.basename(classes_txt_path)))

    json_dict = {'classes': classes}
    with open(os.path.join(output_dir_path, 'classes.json'), 'w') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik-mnist-video-count-dataset')
    parser.add_argument('--classes_txt_path', type=str, default=os.path.join(os.path.dirname(__file__), 'classes.txt'))
    parser.add_argument('--train_sample_num', type=int, default=40000)
    parser.add_argument('--valid_sample_num', type=int, default=100)
    parser.add_argument('--image_max_size', type=int, default=320)
    parser.add_argument('--image_min_size', type=int, default=196)
    parser.add_argument('--char_max_size', type=int, default=64)
    parser.add_argument('--char_min_size', type=int, default=32)
    parser.add_argument('--char_max_num', type=int, default=10)
    parser.add_argument('--char_min_num', type=int, default=1)
    parser.add_argument('--max_delay_frame', type=int, default=8)
    parser.add_argument('--min_frame_num', type=int, default=16)
    parser.add_argument('--max_frame_num', type=int, default=32)
    parser.add_argument('--min_char_speed', type=int, default=-10)
    parser.add_argument('--max_char_speed', type=int, default=10)
    parser.add_argument('--min_char_size_ratio', type=int, default=0.98)
    parser.add_argument('--max_char_size_ratio', type=int, default=1.02)
    parser.add_argument('--occlusion_ratio', type=int, default=0.05)
    args = parser.parse_args()

    args.classes_txt_path = os.path.expanduser(args.classes_txt_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    main(**args.__dict__)
