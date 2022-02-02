import glob
from itertools import compress
import logging
import pathlib

from .datautils import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import scipy.io as sio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class PriceTagDataset(Dataset):
    def __init__(self, data_root, image_ext='jpg', json_ext='json', input_size=256):
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

        image_file, image, score_map, training_mask, transcription, bbox = \
                self.__transform((image_file, bbox, transcription))

        if visualize:
            # print('image_file:', image_file)
            # print('images', image.shape)
            # print('transcriptions', transcription)
            # print('bboxes', bbox)
            # print('score_maps', score_map.shape)
            # print('geo_maps', geo_map.shape)
            # print('training_masks', training_mask.shape)
            transformed_image_filename = image_filename_wo_ext + '_transformed.' + ext
            transformed_image_file = os.path.join(self.visualization_dir, transformed_image_filename)
            cv2.imwrite(transformed_image_file, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            self.visualize(transformed_image_file, bbox, transcription[0])

        return image_file, image, score_map, training_mask, transcription, bbox

    def __len__(self):
        return len(self.image_files)

    def __transform(self, gt, random_scale=[0.5, 1, 2.0, 3.0]):
        image_file, bbox, transcription = gt
        image = cv2.imread(image_file)
        h, w = image.shape[:2]

        ## Renormalize bbox
        bbox *= np.array([w, h, w, h])
        x1, y1, x2, y2 = bbox

        ## Random crop without losing bbox
        top_crop = int(np.random.random() * y1)
        left_crop = int(np.random.random() * x1)
        bottom_crop = int(np.random.random() * (h - y2))
        right_crop = int(np.random.random() * (w - x2))
        image = image[top_crop : h - bottom_crop, left_crop : w - right_crop, :]
        bbox = np.array([x1 - left_crop, y1 - top_crop, x2 - left_crop, y2 - top_crop])

        # Random scale
        rd_scale = np.random.choice(random_scale)
        image = cv2.resize(image, dsize=None, fx=rd_scale, fy=rd_scale, interpolation=cv2.INTER_NEAREST)
        bbox *= rd_scale

        ## Resize image
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
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-10)

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
        x1, y1 = bbox[:2]
        x2, y2 = bbox[2:4]
        x3, y3 = bbox[4:6]
        x4, y4 = bbox[6:]
        image_draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], outline='red')
        image_draw.text((x1, y1), transcription, fill='red', font=font)
        image.save(os.path.join(self.visualization_dir, os.path.basename(image_file)))
