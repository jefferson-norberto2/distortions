
from distortions.test import test_model
from distortions.train import train_model

if __name__ == '__main__':
    model = 'resnet_152'
    # model_path = train_model(backbone=model, num_epochs=20)
    test_model(weight_path='best_distortions_17_resnet_152_b16_lr1e-4.pth', name_model=model)




