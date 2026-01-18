
from distortions.model.custom_resnet import ModelArchitecture
from distortions.test import test_model
from distortions.train import train_model
from distortions.dataset import download


if __name__ == '__main__':
    # Download examples dataset
    URL_LIVE = 'https://drive.google.com/file/d/12cCCuaH7CBcx3VzEeFp8wbW-WskO4m0G/view?usp=drive_link'
    URL_CSIQ = 'https://drive.google.com/file/d/1dfE88U28ntT41EuraCH7gQh4ksAJCFVY/view?usp=drive_link'

    download.download_file(URL_LIVE, './data/LIVE.zip', unzip=True)
    download.download_file(URL_CSIQ, './data/ECSIQ.zip', unzip=True)

    backbone_choice = ModelArchitecture.INCEPTION_V3
    best_model_path = train_model(
        backbone=backbone_choice, 
        data_dir='./data/ECSIQ/',
        num_epochs=10, 
        lr=0.0001,
        batch_size=16,
        wandb_enable=True
    )

    test_model(
        model_path=best_model_path,
        backbone=backbone_choice,
        data_dir='./data/LIVE/',
        batch_size=16,
        wandb_enable=True
    )


