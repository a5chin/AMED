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
    def __init__(self, root: str=config.DATASET.ROOT):
        self.root = Path(root) / 'DICOM'
        self.path = [p for p in self.root.iterdir()]
        self.path_images, self.path_annotations = Path(config.DATASET.IMAGES), Path(config.DATASET.ANNOTATIONS)
        self.path_images.mkdir(exist_ok=True)
        self.path_annotations.mkdir(exist_ok=True)
        self.id = 0
        self.hashes = []
        self.info = []

    def organize(self):
        for p in tqdm(self.path):
            self._read_data(p)


    def _read_data(self, path: Path):
        for p in (path / 'DICOM').glob('*.dcm'):
            ds = dcmread(p)
            image = Image.fromarray(ds.pixel_array)
            info = pd.read_csv(str(path / path.stem) + '.csv', header=None, encoding='shift-jis').values[0].tolist()
            height, width = image.size
            if height > 400 and width > 400:
                _hash = phash(image)
                if not self._in_same_image(_hash):
                    _id = self.id
                    self.id += 1
                    annotations = (p.parents[1] / 'CUTIMAGE').glob('*.csv')
                    for annotation in annotations:
                        if str(p.stem) in str(annotation):
                            df = pd.read_csv(annotation, header=None).to_numpy()[0]
                            points = self._calc_point(df)
                            self.info.append(
                                {'id': _id, 'diagnosis': info[13], 'age': info[8], 'sex': 9, 'points': points, 'height': height, 'width': width}
                            )
                    self.hashes.append(_hash)
                    self._remove_noises(image)
                    # image.save(self.path_images / (str(_id).zfill(5) + '.png'))

    def _in_same_image(self, _hash: imagehash.ImageHash) -> bool:
        return sum(1 for it in self.hashes if it - _hash < 2) != 0

    def _remove_noises(self, image: Image.Image):
        img = np.asarray(image)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = self._make_mask(hsv)

        # TODO
        h_target, w_target = np.where(mask > 0)

        roop = [(j, i)
                for j in range(-2, 2)
                for i in range(-2, 2)]
        for h, w in zip(h_target, w_target):
            for j, i in roop:
                h_min = h + j - 3
                h_max = h + j + 3
                w_min = w + i - 3
                w_max = w + i + 3
                med_val = np.median(img[h_min:h_max, w_min:w_max])
                hsv[h + j, w + i] = med_val + random.randint(-3, 3)

        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        cv2.imshow('sample', img)
        cv2.waitKey(0)

        return hsv

    def _make_mask(self, image: np.ndarray) -> np.ndarray:
        blue_l, blue_u = np.array([70, 20, 20]), np.array([100, 255, 255])
        yellow_l, yellow_u = np.array([20, 20, 40]), np.array([40, 255, 255])

        mask_blue = cv2.inRange(image, blue_l, blue_u)
        mask_yellow = cv2.inRange(image, yellow_l, yellow_u)
        mask = cv2.bitwise_or(mask_blue, mask_yellow)
        return mask


    def _calc_point(self, array: list):
        half = array[2] / 2
        return [array[0] - half, array[1] - half, array[0] + half, array[1] + half]
