import sys

from dotenv import load_dotenv
from distortions.scripts.test.test_mobilenet import run_mobilenet_tests
from distortions.scripts.test.test_resnet import run_resnet_tests
from distortions.scripts.test.test_late import run_late_fusion_tests
from distortions.scripts.test.test_early import run_early_fusion_tests
from distortions.scripts.test.test_yolo_accuracy import run_yolo_accuracy_tests
from distortions.scripts.test.test_yolo_hardware import run_yolo_hardware_tests


load_dotenv()

if __name__ == "__main__":
    selected_test = ""

    if len(sys.argv) > 1:
        selected_test = sys.argv[1].lower()
    
    if selected_test == "resnet":
        run_resnet_tests()
    elif selected_test == "late":
        run_late_fusion_tests()
    elif selected_test == "early":
        run_early_fusion_tests()
    elif selected_test == "yolo_accuracy":
        run_yolo_accuracy_tests()
    elif selected_test == "yolo_hardware":
        run_yolo_hardware_tests()
    elif selected_test == "mobilenet":
        run_mobilenet_tests()
    else:
        raise ValueError(f"Invalid test selection: {selected_test}. \nPlease choose from 'resnet', 'late', 'early', 'yolo_accuracy', 'yolo_hardware', or 'mobilenet'.")