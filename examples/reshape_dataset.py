import argparse
import pathlib
import sys
import warnings

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + "/../")
warnings.filterwarnings("ignore")

from amed.dataset import Reshaper


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--save_path",
        default=r"/work/hara.e/AMED/lib/dataset/annotations.json",
        help="please set save path",
        type=str,
    )

    return parser.parse_args()


def main():
    args = make_parser()

    reshaper = Reshaper()
    reshaper.organize(args.save_path)


if __name__ == "__main__":
    main()
