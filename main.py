
from distortions.test import test_model
from distortions.train import train_model

if __name__ == '__main__':
    model = 'resnet_152'
    
    test_model(weight_path='runs/train/20251025-043139/best_distortion_7.pth', name_model=model, folder_path='/home/jmn/host/dev/Datasets/IQA/ECSIQ/')




