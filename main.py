from tools import Trainer
from lib.dataset import Reshaper


def main():
    # trainer = Trainer(model_name='yolov5s')
    reshaper = Reshaper()
    reshaper.organize()
    # dataset = DicomDataset()
    # dataloader = DataLoader(dataset)
    # for image in dataloader:
    #     print('done')


if __name__ == '__main__':
    main()
