from tools import Trainer
from lib.dataset import Reshaper
from lib.dataset import CutOuter


def main():
    trainer = Trainer(model_name='YOLOX')
    trainer.train()


if __name__ == '__main__':
    main()
