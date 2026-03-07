
from distortions.test import test_model
from distortions.train import train_model
from distortions.dataset import download


if __name__ == '__main__':
    # Download examples dataset
    # URL_LIVE = 'https://drive.google.com/file/d/12cCCuaH7CBcx3VzEeFp8wbW-WskO4m0G/view?usp=drive_link'
    # URL_CSIQ = 'https://drive.google.com/file/d/1dfE88U28ntT41EuraCH7gQh4ksAJCFVY/view?usp=drive_link'

    # download.download_file(URL_LIVE, './data/LIVE.zip', unzip=True)
    # download.download_file(URL_CSIQ, './data/ECSIQ.zip', unzip=True)

    # Train and test model
    model = 'resnet18gdn' # 'resnet50', 'resnet101', 'resnet152', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'
    wandb_enable = False
    best = train_model(backbone=model, 
                       data_dir='/home/jmn/Dev/Datasets/Distortions_v4/', 
                       num_epochs=10, 
                       batch_size=16, 
                       lr=0.0001, 
                       wandb_enable=wandb_enable, 
                       img_size=512
                       )
    
    # test_model(
    #     weight_path=best, 
    #     name_model=model, 
    #     folder_path='/home/jmn/Dev/Datasets/LIVE_512', 
    #     wandb_enable=wandb_enable)


