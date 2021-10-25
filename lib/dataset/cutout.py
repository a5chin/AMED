from tqdm import tqdm
from json import load, dump
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split


class CutOuter:
    def __init__(self, root: str=r'\\aka\work\hara.e\AMED\lib\dataset'):
        self.root = Path(root)
        json_path = self.root / r'annotations/temp.json'
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

    def train_test_split(self):
        typ = ('validation', 'test')
        info, licenses, categories = self.data['info'], self.data['licenses'], self.data['categories']
        data = {
            t: {'info': info, 'licenses': licenses, 'images': {}, 'annotations': {}, 'categories': categories}
            for t in typ
        }
        images = [image for image in self.data['images']]
        annotations = [annotations for annotations in self.data['annotations']]
        images_train, images_test, annotations_train, annotations_test \
            = train_test_split(images, annotations, test_size=0.5)
        data[typ[0]]['images'], data[typ[0]]['annotations'] \
            = sorted(images_train, key=lambda x: x['id']), sorted(annotations_train, key=lambda x: x['id'])
        data[typ[1]]['images'], data[typ[1]]['annotations'] \
            = sorted(images_test, key=lambda x: x['id']), sorted(annotations_test, key=lambda x: x['id'])
        for t in typ:
            with open(self.root / f'annotations/{t}.json', 'w') as f:
                dump(data[t], f)
