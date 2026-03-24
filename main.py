
from distortions.train import train_model


if __name__ == '__main__':
    # Train and test model
    models = ['resnet152', 'resnet101', 'resnet50', 'resnet34']
    wandb_enable = False

    for model in models:
        try:
            train_model(backbone=model, 
                            data_dir='Datasets/LIST/', 
                            num_epochs=20, 
                            batch_size=16, 
                            lr=0.0001, 
                            wandb_enable=wandb_enable, 
                            img_size=512
                        )
        except Exception as e:
            print(f"Error training {model}: {e}")


