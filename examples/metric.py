import argparse
import json
import pathlib
import sys
import warnings
from typing import Dict

import numpy as np
import pandas as pd

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + "/../")
warnings.filterwarnings("ignore")

from amed.utils import Metric, calc_iou, get_logger


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_file",
        default=r"//aka/work/hara.e/AMED/lib/dataset/annotations/test.json",
        type=str,
        help="please set label file",
    )
    parser.add_argument(
        "--pred_file",
        default=r"//aka/work/hara.e/AMED/weights/CenterNet/four/metric0.3.json",
        type=str,
        help="please set prediction file",
    )

    return parser.parse_args()


def load_json(file_name: str) -> Dict:
    with open(file_name, "r") as f:
        dct = json.load(f)
    return dct


def create_df_label(dct: Dict) -> pd.DataFrame:
    df = {key: [] for key in dct[0].keys()}

    for d in dct:
        for key in df.keys():
            df[key].append(d[key])

    return pd.DataFrame(df)


def create_df_pred(dct: Dict) -> pd.DataFrame:
    df = {key: [] for key in dct.keys()}

    for pred in zip(*(dct[key] for key in dct.keys())):
        for i, key in enumerate(dct.keys()):
            df[key].append(pred[i])

    return pd.DataFrame(df)


def create_df_iou(preds: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    df_ious = {"ious": []}

    for i in range(len(preds)):
        iou = calc_iou(preds.iloc[i]["bbox"], labels.iloc[i]["bbox"])
        df_ious["ious"].append(iou)

    return pd.DataFrame(df_ious)


def preprocess(args) -> pd.DataFrame:
    labels = load_json(args.label_file)["annotations"]
    preds = load_json(args.pred_file)

    df_label = create_df_label(labels)
    df_preds = create_df_pred(preds)
    df_preds["category_id"] = df_label["category_id"]

    df_ious = create_df_iou(df_preds, df_label)
    df_preds["ious"] = df_ious

    return df_preds


def main():
    args = make_parser()
    logger = get_logger()

    metric = Metric(conf_th=0.3)
    cmat = np.zeros((5, 5))

    df = preprocess(args)
    for idx in range(len(df)):
        row = df.iloc[idx, :]
        cmat += metric.update(row)

    precision, recall, f1 = metric.evaluate(cmat)
    logger.info(f"precision: {precision}, recall: {recall}, f1: {f1}")


if __name__ == "__main__":
    main()
