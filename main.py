
from distortions.test import test_model
from distortions.train import train_model

if __name__ == '__main__':
    model = 'resnet_152'
    model_path = train_model(backbone=model)
    test_model(weight_path=model_path, name_model=model)




