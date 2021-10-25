from tqdm import tqdm
from json import load, dump
from pathlib import Path
from PIL import Image


class CutOuter:
    def __init__(self, root: str=r'\\aka\work\hara.e\AMED\lib\dataset'):
        self.root = Path(root)
        json_path = self.root / r'annotations/annotations.json'
        with open(json_path) as f:
            self.data = load(f)

    def cut_out(self):
        root = self.root / 'cutout'
        for image, annotation in tqdm(zip(self.data['images'], self.data['annotations'])):
            file_name = image['file_name']
            bbox = annotation['bbox']
            _id = image['id']
            img = Image.open(self.root / 'images' / file_name)
            img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])).save(root / 'images' / file_name)
            self.data['images'][_id]['height'], self.data['images'][_id]['width'] = bbox[2], bbox[3]
        with open(root / 'annotations/annotations.json', 'w') as f:
            dump(self.data, f)
