from yacs.config import CfgNode as CN

_C = CN()

_C.DATASET = CN()
_C.DATASET.NAME = 'AMED'
_C.DATASET.ROOT = r'\\aka\work\hara.e\AMED\lib\dataset'
_C.DATASET.IMAGES = _C.DATASET.ROOT + r'\images'
_C.DATASET.ANNOTATIONS = _C.DATASET.ROOT + r'\annotations'
_C.DATASET.BATCH_SIZE = 1


def get_defaults():
    return _C.clone()


def load_config(config_path):
    cfg = get_defaults()
    cfg.merge_from_file(config_path)
    cfg.freeze()
    return cfg
