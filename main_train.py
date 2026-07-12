import sys

from dotenv import load_dotenv
from distortions.scripts.train.train_mobilenet import train_mobilenets
from distortions.scripts.train.train_resnet import train_resnets
from distortions.scripts.train.train_early import train_early_fusion
from distortions.scripts.train.train_late import train_late_fusion
from distortions.scripts.train.train_yolo import train_yolos

load_dotenv()

if __name__ == "__main__":
    selected_train = "mobilenet"

    if len(sys.argv) > 1:
        selected_train = sys.argv[1].lower()
    
    if selected_train == "resnet":
        train_resnets()
    elif selected_train == "late":
        train_late_fusion()
    elif selected_train == "early":
        train_early_fusion()
    elif selected_train == "yolo":
        train_yolos()
    elif selected_train == "mobilenet":
        train_mobilenets()
    else:
        raise ValueError(f"Invalid training selection: {selected_train}. \nPlease choose from 'resnet', 'late', 'early', 'yolo', or 'mobilenet'.")