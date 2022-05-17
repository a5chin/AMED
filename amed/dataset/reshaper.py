import numpy as np
import pandas as pd
import cv2
import random
import imagehash
import json
from imagehash import phash
from pydicom import dcmread
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from lib.config import config


class Reshaper:
    def __init__(self, root: str = config.DATASET.ROOT):
        self.root = Path(root) / 'DICOM'
        self.path = [p for p in self.root.iterdir()]
        self.images_path, self.annotations_path = Path(config.DATASET.IMAGES), Path(config.DATASET.ANNOTATIONS)
        self.images_path.mkdir(exist_ok=True)
        self.annotations_path.mkdir(exist_ok=True)
        self.id = 0
        self.hashes = []
        self.name = ['単純嚢胞', '肝細胞癌', '血管腫', '転移性肝癌']
        self.info = {
            'info': {
                'description': 'AMED Dataset',
                'url': 'https://www.amed.go.jp',
                'version': 1.0,
                'year': 2021,
                'contributor': 'Japan Agency for Medical Research and Development',
                'data_created': '2021/10/22'
            },
            'licenses': [{
                'url': 'https://www.amed.go.jp',
                'id': 1,
                'name': 'Japan Agency for Medical Research and Development'
            }],
            'images': [], 'annotations': [],
            'categories': [
                {
                    'supercategory': 'cancer',
                    'id': 1,
                    'name': 'cyst'
                },
                {
                    'supercategory': 'cancer',
                    'id': 2,
                    'name': 'hcc'
                },
                {
                    'supercategory': 'cancer',
                    'id': 3,
                    'name': 'hemangioma'
                },
                {
                    'supercategory': 'cancer',
                    'id': 4,
                    'name': 'meta'
                },
                {
                    'supercategory': 'cancer',
                    'id': 5,
                    'name': 'other'
                },
            ]
        }

    def organize(self):
        for p in tqdm(self.path):
            self._read_data(p)

        self._create_json()

    def _read_data(self, path: Path):
        for p in (path / 'DICOM').glob('*.dcm'):
            try:
                ds = dcmread(p)
                image = Image.fromarray(ds.pixel_array)
                info = pd.read_csv(str(path / path.stem) + '.csv', header=None, encoding='shift-jis').values[0].tolist()
                width, height = image.size
                if height > 400 and width > 400 and info[13] in self.name:
                    _hash = phash(image)
                    if not self._in_same_image(_hash):
                        annotations = (p.parents[1] / 'CUTIMAGE').glob('*.csv')
                        for annotation in annotations:
                            if str(p.stem) in str(annotation):
                                df = pd.read_csv(annotation, header=None).to_numpy()[0]
                                points = self._calc_point(df)

                                self.info['images'].append({
                                    'licenses': 1,
                                    'file_name': str(self.id).zfill(6) + '.jpg',
                                    'height': height,
                                    'width': width,
                                    'id': self.id
                                })
                                self.info['annotations'].append({
                                    'area': width * height,
                                    'iscrowd': 0,
                                    'image_id': self.id,
                                    'bbox': points,
                                    'category_id': self._convert_diagnosis(info[13]),
                                    'id': self.id,
                                    'age': info[8],
                                    'sex': 1 if ds.PatientSex == 'M' else 0
                                })

                                self.hashes.append(_hash)
                                image = self._remove_noises(image)
                                image.save(self.images_path / (str(self.id).zfill(6) + '.jpg'))
                                self.id += 1
                                if self.id % 5000 == 0:
                                    with open(f'./annotations{self.id}.json', 'w') as f:
                                        print(self.id)
                                        json.dump(self.info, f)
            except:
                pass

    def _in_same_image(self, _hash: imagehash.ImageHash) -> bool:
        return sum(1 for it in self.hashes if it - _hash < 2) != 0

    def _remove_noises(self, image: Image.Image, kernel_size: int=3, random_range: tuple=(-3, 3)) -> Image.Image:
        img = np.asarray(image, dtype=np.uint8)

        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        except:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = self._make_mask(hsv)
        height, width = mask.shape
        x_target, y_target = np.where(mask > 0)

        for x, y in zip(x_target, y_target):
            x_min, x_max = x - kernel_size, x + kernel_size
            y_min, y_max = y - kernel_size, y + kernel_size
            if x_min < 0 or width < x_max or y_min < 0 or height < y_max:
                continue
            img[x, y] = np.median(img[x_min: x_max, y_min: y_max]) + random.randint(*random_range)

        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _make_mask(image: np.ndarray) -> np.ndarray:
        blue_l, blue_u = np.array([70, 20, 20]), np.array([100, 255, 255])
        yellow_l, yellow_u = np.array([20, 20, 40]), np.array([40, 255, 255])

        mask_blue = cv2.inRange(image, blue_l, blue_u)
        mask_yellow = cv2.inRange(image, yellow_l, yellow_u)
        mask = cv2.bitwise_or(mask_blue, mask_yellow)
        return mask

    @staticmethod
    def _calc_point(array: list) -> list:
        half = array[2] / 2
        return [array[0] - half, array[1] - half, half * 2, half * 2]

    def _create_json(self):
        with open(self.annotations_path / 'annotations.json', 'w') as f:
            json.dump(self.info, f)

    def _convert_diagnosis(self, diagnosis: str):
        if diagnosis in self.name:
            diagnosis = self.name.index(diagnosis) + 1
        else:
            diagnosis = 5
        return diagnosis
