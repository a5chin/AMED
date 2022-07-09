import sys
from pathlib import Path

import numpy as np

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from amed.utils import Metric


def test1():
    df = {
        "file_name": "000000.jpg",
        "preds": [1],
        "reliability": [0.6959085464477539],
        "bbox": [[344.4757995605469, 434.115478515625, 464.0614318847656, 554.0039672851562]],
        "category_id": [1],
        "ious": [0.5330510139465332],
    }

    correct = np.zeros((5, 5)).astype(np.int16)
    correct[1, 1] += 1
    metric = Metric(conf_th=0.35)

    metric.update(df)
    cmat = metric.cmat

    assert np.array_equal(cmat, correct)


def test2():
    df = {
        "file_name": "000020.jpg",
        "preds": [4],
        "reliability": [0.48194679617881775],
        "bbox": [[243.02191162109375, 390.07061767578125, 412.15081787109375, 558.7544555664062]],
        "category_id": [4],
        "ious": [0.08640196174383163],
    }

    correct = np.zeros((5, 5)).astype(np.int16)
    correct[0, 4] += 1
    correct[4, 0] += 1
    metric = Metric(conf_th=0.35)

    metric.update(df)
    cmat = metric.cmat

    assert np.array_equal(cmat, correct)


def test3():
    df = {
        "file_name": "0000104.jpg",
        "preds": [1, 1],
        "reliability": [0.7432386875152588, 0.5572695732116699],
        "bbox": [
            [296.0484924316406, 455.67510986328125, 379.3522644042969, 538.9735107421875],
            [374.1202392578125, 217.28932189941406, 463.3363037109375, 306.39288330078125],
        ],
        "category_id": [1],
        "ious": [0.6415560841560364, 0.0],
    }

    correct = np.zeros((5, 5)).astype(np.int16)
    correct[0, 1] += 1
    correct[1, 1] += 1
    metric = Metric(conf_th=0.35)

    metric.update(df)
    cmat = metric.cmat

    assert np.array_equal(cmat, correct)


def test4():
    df = {
        "file_name": "006611.jpg",
        "preds": [1, 1],
        "reliability": [0.7432386875152588, 0.5572695732116699],
        "bbox": [
            [283.68603515625, 217.69776916503906, 345.06884765625, 279.2651672363281],
            [623.0662231445312, 219.28660583496094, 683.1857299804688, 278.92523193359375],
        ],
        "category_id": [3],
        "ious": [0.0029819835908710957, 0.0],
    }

    correct = np.zeros((5, 5)).astype(np.int16)
    correct[0, 1] += 2
    correct[3, 0] += 1
    metric = Metric(conf_th=0.35)

    metric.update(df)
    cmat = metric.cmat

    assert np.array_equal(cmat, correct)


def test5():
    df = {"file_name": "007234.jpg", "preds": [0], "reliability": [0.0], "bbox": [[]], "category_id": [3], "ious": [0.0]}

    correct = np.zeros((5, 5)).astype(np.int16)
    correct[3, 0] += 1
    metric = Metric(conf_th=0.35)

    metric.update(df)
    cmat = metric.cmat

    assert np.array_equal(cmat, correct)


def test6():
    df = {
        "file_name": "008788.jpg",
        "preds": [1, 4],
        "reliability": [0.42805975675582886, 0.3644619584083557],
        "bbox": [
            [457.1584167480469, 117.6557388305664, 513.4650268554688, 174.31973266601562],
            [457.1584167480469, 117.6557388305664, 513.4650268554688, 174.31973266601562],
        ],
        "category_id": [1],
        "ious": [0.7789446711540222, 0.7789446711540222],
    }

    correct = np.zeros((5, 5)).astype(np.int16)
    correct[1, 1] += 1
    metric = Metric(conf_th=0.35)

    metric.update(df)
    cmat = metric.cmat

    assert np.array_equal(cmat, correct)


def test7():
    df = {
        "file_name": "026101.jpg",
        "preds": [1, 1],
        "reliability": [0.5039288401603699, 0.36427226662635803],
        "bbox": [
            [419.8945007324219, 231.20445251464844, 495.8955383300781, 307.16522216796875],
            [396.04669189453125, 261.0216064453125, 455.3426513671875, 320.08624267578125],
        ],
        "category_id": [1],
        "ious": [0.0, 0.0],
    }

    correct = np.zeros((5, 5)).astype(np.int16)
    correct[1, 0] += 1
    correct[0, 1] += 2
    metric = Metric(conf_th=0.35)

    metric.update(df)
    cmat = metric.cmat

    assert np.array_equal(cmat, correct)
