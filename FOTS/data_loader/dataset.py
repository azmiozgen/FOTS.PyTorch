import glob
import json
import logging
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

from .datautils import (clip_box, denormalize, get_corners, get_enclosing_box,
        rotate_image, rotate_box)

logger = logging.getLogger(__name__)

class PriceTagDataset(Dataset):
    def __init__(self, data_root, train_mode=True, image_ext='jpg', json_ext='json', input_size=256):
        images_dir = os.path.join(data_root, 'images')
        annotations_dir = os.path.join(data_root, 'annotations')
        self.image_files = sorted(glob.glob(os.path.join(images_dir, '*.' + image_ext)))
        self.annotation_files = sorted(glob.glob(os.path.join(annotations_dir, '*.' + json_ext)))
        self.image_filenames = list(map(lambda x: os.path.basename(x), self.image_files))
        self.bboxes, self.transcriptions = [], []
        for ann_file in self.annotation_files:
            bbox, transcription = self._load_annotation(ann_file)
            self.bboxes.append(bbox)
            self.transcriptions.append(transcription)
        self.visualization_dir = os.path.join(data_root, 'visualization')
        os.makedirs(self.visualization_dir, exist_ok=True)
        self.input_size = input_size
        self.train_mode = train_mode

    def _load_annotation(self, ann_file, keys=['price', 'price_cent'],
                bbox_name='bbox', content_name='content', delimiter=','):
        bbox = []
        transcription = ''

        if not os.path.isfile(ann_file):
            return bbox, transcription

        with open(ann_file, 'r') as f:
            j = json.load(f)

        bbox = [1.0, 1.0, 0.0, 0.0]
        for key in keys:
            _bbox = j[key][bbox_name]
            if len(_bbox) > 0:
                _x0, _y0, _x1, _y1 = _bbox
                bbox = [min(_x0, bbox[0]), min(_y0, bbox[1]), max(_x1, bbox[2]), max(_y1, bbox[3])]
                content = j[key][content_name]
                if key == 'price_cent' and content != '':
                    transcription += delimiter
                transcription += content

        _x0, _y0, _x1, _y1 = bbox
        if _x0 > _x1:
            _x0, _x1 = _x1, _x0
        if _y0 > _y1:
            _y0, _y1 = _y1, _y0
        x0, y0, x1, y1 = max(0, _x0), max(0, _y0), max(0, _x1), max(0, _y1)
        x0, y0, x1, y1 = min(1, _x0), min(1, _y0), min(1, _x1), min(1, _y1)
        bbox = [x0, y0, x1, y1]

        return bbox, transcription

    def __getitem__(self, index, visualize=False):
        '''
        :param: index
        :return:
            image_file: path of image file
            bbox: bounding box of word
            transcription: transcription of word
        '''
        image_file = self.image_files[index]
        image_filename = self.image_filenames[index]
        image_filename_wo_ext, ext = os.path.splitext(image_filename)
        bbox = self.bboxes[index]
        transcription = self.transcriptions[index]

        if visualize:
            self.visualize(image_file, bbox, transcription)

        image_file, image, score_map, training_mask, transcription, bbox = \
                self.__transform((image_file, bbox, transcription))

        if visualize:
            transformed_image_filename = image_filename_wo_ext + '_transformed.' + ext
            transformed_image_file = os.path.join(self.visualization_dir, transformed_image_filename)
            image = denormalize(image, from_min=-1, from_max=0, to_min=0, to_max=255)
            cv2.imwrite(transformed_image_file, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            self.visualize(transformed_image_file, bbox, transcription[0])

        return image_file, image, score_map, training_mask, transcription, bbox

    def __len__(self):
        return len(self.image_files)

    def __transform(self, gt,
            random_scales=[0.5, 1, 2.0, 3.0],
            random_rotation_angles=[-10, 11]):
        image_file, bbox, transcription = gt
        image = cv2.imread(image_file)
        h, w = image.shape[:2]

        ## Renormalize bbox
        bbox *= np.array([w, h, w, h])
        x1, y1, x2, y2 = bbox

        ## Random crop without losing bbox
        if self.train_mode:
            top_crop = int(np.random.random() * y1)
            left_crop = int(np.random.random() * x1)
            bottom_crop = int(np.random.random() * (h - y2))
            right_crop = int(np.random.random() * (w - x2))
            image = image[top_crop : h - bottom_crop, left_crop : w - right_crop, :]
            bbox = np.array([x1 - left_crop, y1 - top_crop, x2 - left_crop, y2 - top_crop])

            # Random scale
            rd_scale = np.random.choice(random_scales)
            image = cv2.resize(image, dsize=None, fx=rd_scale, fy=rd_scale, interpolation=cv2.INTER_NEAREST)
            bbox *= rd_scale

            ## Random rotation
            rotation_angle = np.random.randint(*random_rotation_angles)
            image, bboxes = RandomRotate(angle=rotation_angle)(image, bbox.reshape(1, -1))
            bbox = bboxes[0]

        ## Resize image to input size
        _h, _w, _ = image.shape
        image = cv2.resize(image, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        ratio_w = self.input_size / _w
        ratio_h = self.input_size / _h
        bbox *= np.array([ratio_w, ratio_h, ratio_w, ratio_h])

        ## Set score map, geo map and training mask
        score_map = np.zeros((self.input_size, self.input_size), dtype=np.uint8)
        training_mask = np.ones((self.input_size, self.input_size), dtype=np.uint8)  ## One region

        ## Mask score map by bbox
        _bbox = bbox.astype(np.int32)
        score_map[_bbox[1]:_bbox[3], _bbox[0]:_bbox[2]] = 1

        ## Convert bbox to 8-points
        _x0, _y0, _x1, _y1 = bbox
        x1, y1, x2, y2, x3, y3, x4, y4 = _x0, _y0, _x1, _y0, _x1, _y1, _x0, _y1
        bbox = [x1, y1, x2, y2, x3, y3, x4, y4]

        ## Normalize image
        # image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-10)           ## [0, 1]
        image = 2 * ((image - np.min(image)) / (np.max(image) - np.min(image) + 1e-10)) - 1   ## [-1, 1]

        ## Arrange
        images = image[:, :, ::-1].astype(np.float32)  # bgr -> rgb
        score_maps = score_map[::4, ::4, np.newaxis].astype(np.float32)
        training_masks = training_mask[::4, ::4, np.newaxis].astype(np.float32)
        transcription = np.array([transcription], dtype=str)

        return image_file, images, score_maps, training_masks, transcription, bbox

    def visualize(self, image_file, bbox, transcription):
        bbox = np.array(bbox).astype(int)
        image = Image.open(image_file)
        image_draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('FreeMono.ttf', 20)
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            image_draw.polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], outline='red')
            image_draw.text((x1, y1), transcription, fill='red', font=font)
        elif len(bbox) == 8:
            x1, y1 = bbox[:2]
            x2, y2 = bbox[2:4]
            x3, y3 = bbox[4:6]
            x4, y4 = bbox[6:]
            image_draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], outline='red')
            image_draw.text((x1, y1), transcription, fill='red', font=font)
        image.save(os.path.join(self.visualization_dir, os.path.basename(image_file)))

class RandomRotate(object):
    """Randomly rotates an image    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    Parameters
    ----------
    angle: float or tuple(float)
        if **float**, the image is rotated by a factor drawn 
        randomly from a range (-`angle`, `angle`). If **tuple**,
        the `angle` is drawn randomly from values specified by the 
        tuple
    Returns
    -------
    numpy.ndarray
        Rotated image in the numpy format of shape `HxWxC`
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
    """

    def __init__(self, angle=10):
        self.angle = angle
        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"
        else:
            self.angle = (-self.angle, self.angle)

    def __call__(self, img, bboxes):
        angle = random.uniform(*self.angle)

        w, h = img.shape[1], img.shape[0]
        cx, cy = w // 2, h // 2

        img = rotate_image(img, angle)
        corners = get_corners(bboxes)
    
        corners = np.hstack((corners, bboxes[:, 4:]))
        corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)
        new_bbox = get_enclosing_box(corners)

        scale_factor_x = img.shape[1] / w
        scale_factor_y = img.shape[0] / h
        img = cv2.resize(img, (w, h))

        new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
        bboxes  = new_bbox
        bboxes = clip_box(bboxes, [0, 0, w, h], 0.25)

        return img, bboxes
