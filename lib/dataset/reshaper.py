import pandas as pd
from imagehash import phash
from pydicom import dcmread
from PIL import Image
from pathlib import Path

from lib.config import config


class Reshaper:
    cnt = 0

    def __init__(self, root: str=config.DATASET.ROOT):
        self.root = Path(root) / 'DICOM'
        self.path = [p for p in self.root.iterdir()]
        self.path_images, self.path_annotations = Path(config.DATASET.IMAGES), Path(config.DATASET.ANNOTATIONS)
        self.path_images.mkdir(exist_ok=True)
        self.path_annotations.mkdir(exist_ok=True)
        self.hashes = []

    def organize(self):
        for p in self.path:
            self._read_data(p)

    def _read_data(self, path: Path):
        for p in (path / 'DICOM').glob('*.dcm'):
            ds = dcmread(p)
            image = Image.fromarray(ds.pixel_array)
            _hash = phash(image)
            if sum(1 for it in self.hashes if it - _hash < 2) == 0:
                _id = Reshaper.cnt
                Reshaper.cnt += 1
                annotations = (p.parents[1] / 'CUTIMAGE').glob('*.csv')
                for annotation in annotations:
                    if str(p.stem) in str(annotation):
                        df = pd.read_csv(annotation, header=None).to_numpy()[0]
                        print(df)
                self.hashes.append(_hash)
                # image.save(self.path_images / (str(_id).zfill(5) + '.png'))
