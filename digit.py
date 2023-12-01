import random
from PIL import Image
import numpy as np
class Digit:
    def __init__(self, org_image, label, canvas_shape, min_speed, max_speed, char_min_size, char_max_size, min_char_size_ratio, max_char_size_ratio, occlusion_ratio, max_delay_frame, size_change_ratio=0.1, speed_change_ratio=0.1):
        self.current_color_image, self.current_mask_image = self.__init_org_image(org_image, char_min_size, char_max_size)
        self.canvas_shape = canvas_shape
        self.label = label
        self.center_x = random.randint(0, canvas_shape[1]-1)
        self.center_y = random.randint(0, canvas_shape[0]-1)
        self.min_speed, self.max_speed = min_speed, max_speed
        self.delay_frame = random.randint(0, max_delay_frame)
        self.current_x_speed = random.randint(self.min_speed, self.max_speed)
        self.current_y_speed = random.randint(self.min_speed, self.max_speed)
        self.min_char_size_ratio, self.max_char_size_ratio = min_char_size_ratio, max_char_size_ratio
        self.current_x_size = random.uniform(self.min_char_size_ratio, self.max_char_size_ratio)
        self.current_y_size = random.uniform(self.min_char_size_ratio, self.max_char_size_ratio)
        self.occlusion_ratio = occlusion_ratio
        self.occlusion_flag = False
        self.size_change_ratio = size_change_ratio
        self.speed_change_ratio = speed_change_ratio
        self.frame_index = 0

    def paste(self, canvas):
        if self.occlusion_flag or self.frame_index < self.delay_frame:
            self.__next()
            return canvas, False

        paste_start_x = self.center_x - self.current_color_image.shape[1]//2
        if paste_start_x < 0:
            crop_start_x = -paste_start_x
            paste_start_x = 0
        else:
            crop_start_x = 0
        paste_start_y = self.center_y - self.current_color_image.shape[0]//2
        if paste_start_y < 0:
            crop_start_y = -paste_start_y
            paste_start_y = 0
        else:
            crop_start_y = 0
        paste_end_x = paste_start_x + self.current_color_image.shape[1] - crop_start_x
        if paste_end_x > self.canvas_shape[1]:
            crop_end_x = self.current_color_image.shape[1] - (paste_end_x - self.canvas_shape[1])
            paste_end_x = self.canvas_shape[1]
        else:
            crop_end_x = self.current_color_image.shape[1]
        paste_end_y = paste_start_y + self.current_color_image.shape[0] - crop_start_y
        if paste_end_y > self.canvas_shape[0]:
            crop_end_y = self.current_color_image.shape[0] - (paste_end_y - self.canvas_shape[0])
            paste_end_y = self.canvas_shape[0]
        else:
            crop_end_y = self.current_color_image.shape[0]
        paste_image = self.current_color_image[crop_start_y:crop_end_y, crop_start_x:crop_end_x, :]
        paste_mask_image = self.current_mask_image[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
        canvas[paste_start_y:paste_end_y, paste_start_x:paste_end_x, :][paste_mask_image > 0] = \
            paste_image[paste_mask_image > 0]
        self.__next()
        return canvas, True

    def __next(self):
        self.frame_index += 1
        if self.frame_index < self.delay_frame:
            return
        self.center_x = self.current_x_speed + self.center_x
        if self.center_x < 0:
            self.center_x = 0
            self.current_x_speed = -self.current_x_speed
        if self.center_x > self.canvas_shape[1]:
            self.center_x = self.canvas_shape[1] - 1
            self.current_x_speed = -self.current_x_speed
        self.center_y = self.current_y_speed + self.center_y
        if self.center_y < 0:
            self.center_y = 0
            self.current_y_speed = -self.current_y_speed
        if self.center_y > self.canvas_shape[0]:
            self.center_y = self.canvas_shape[0] - 1
            self.current_y_speed = -self.current_y_speed

        self.current_color_image = np.array(Image.fromarray(self.current_color_image).resize((
            max(4, int(self.current_color_image.shape[1]*self.current_x_size)),
                max(16, int(self.current_color_image.shape[0]*self.current_y_size)))))
        self.current_mask_image = np.array(Image.fromarray(self.current_mask_image).resize((
            max(4, int(self.current_mask_image.shape[1]*self.current_x_size)),
                max(16, int(self.current_mask_image.shape[0]*self.current_y_size)))))

        if random.uniform(0, 1.0) < self.occlusion_ratio:
            self.occlusion_flag = not self.occlusion_flag

        if random.uniform(0, 1.0) < self.speed_change_ratio:
            self.current_x_speed = random.randint(self.min_speed, self.max_speed)
            self.current_y_speed = random.randint(self.min_speed, self.max_speed)

        if random.uniform(0, 1.0) < self.size_change_ratio:
            self.current_x_size = random.uniform(self.min_char_size_ratio, self.max_char_size_ratio)
            self.current_y_size = random.uniform(self.min_char_size_ratio, self.max_char_size_ratio)

    def __init_org_image(self, org_image, char_min_size, char_max_size):
        mnist_char_image = np.array(Image.fromarray(org_image).resize(
            (random.randint(char_min_size, char_max_size), (random.randint(char_min_size, char_max_size)))))
        mnist_char_image = np.clip((mnist_char_image > 125) * 255, 0, 255)
        mnist_char_image = self.__crop_char_image(mnist_char_image).astype(np.uint8)
        mnist_color_char_image = np.zeros(mnist_char_image.shape + (3,), dtype=np.uint8)
        mnist_color_char_image[:, :, 0] = np.clip((mnist_char_image * random.uniform(0., 1.)).astype(np.uint8), 0,
                                                  255)
        mnist_color_char_image[:, :, 1] = np.clip((mnist_char_image * random.uniform(0., 1.)).astype(np.uint8), 0,
                                                  255)
        mnist_color_char_image[:, :, 2] = np.clip((mnist_char_image * random.uniform(0., 1.)).astype(np.uint8), 0,
                                                  255)
        return mnist_color_char_image, mnist_char_image

    def __crop_char_image(self, mnist_char_image):
        x_min, y_min, x_max, y_max = 0, 0, mnist_char_image.shape[1], mnist_char_image.shape[0]

        x_sum_mnist_char_image = np.sum(mnist_char_image, axis=0)
        y_sum_mnist_char_image = np.sum(mnist_char_image, axis=1)

        for i in range(x_sum_mnist_char_image.shape[0]):
            if x_sum_mnist_char_image[i] > 0:
                x_min = i
                break

        for i in range(y_sum_mnist_char_image.shape[0]):
            if y_sum_mnist_char_image[i] > 0:
                y_min = i
                break

        x_revert_sum_diff_bool_image = x_sum_mnist_char_image[::-1]
        y_revert_sum_diff_bool_image = y_sum_mnist_char_image[::-1]

        for i in range(x_revert_sum_diff_bool_image.shape[0]):
            if x_revert_sum_diff_bool_image[i] > 0:
                x_max = x_revert_sum_diff_bool_image.shape[0] - i
                break

        for i in range(y_revert_sum_diff_bool_image.shape[0]):
            if y_revert_sum_diff_bool_image[i] > 0:
                y_max = y_revert_sum_diff_bool_image.shape[0] - i
                break

        return mnist_char_image[y_min:y_max, x_min:x_max]