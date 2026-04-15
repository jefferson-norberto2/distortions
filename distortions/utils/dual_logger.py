import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix


class DualLogger:
    def __init__(self, args, base_dir="runs/dual_stream", train_dataset=None, val_dataset=None):
        self.args = args
        self.save_dir = self._create_folder(base_dir)
        self.results_file = self.save_dir / "results.csv"
        self.history = []
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self._add_dataset_info()
        self._save_yaml()
        print(f"🚀 Experiment saved in: {self.save_dir}")

    def _create_folder(self, base_dir):
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        existing = [d.name for d in base_path.iterdir() if d.is_dir() and d.name.startswith("train")]
        index = len(existing) + 1
        save_dir = base_path / f"train{index}"
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir
    
    def _add_dataset_info(self):
        if self.train_dataset and self.val_dataset:
            self.args['train_samples'] = len(self.train_dataset)
            self.args['val_samples'] = len(self.val_dataset)
            self.args['total_samples'] = self.args['train_samples'] + self.args['val_samples']
            self.args['num_classes'] = len(self.train_dataset.class_names)
            self.args['class_names'] = self.train_dataset.class_names

    def _save_yaml(self):
        with (self.save_dir / "args.yaml").open('w') as f:
            yaml.dump(self.args, f, default_flow_style=False)

    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc):
        row = {"epoch": epoch, "train/loss": train_loss, "train/accuracy": train_acc, "val/loss": val_loss, 
               "val/accuracy": val_acc, }
        self.history.append(row)
        pd.DataFrame(self.history).to_csv(self.results_file, index=False)
        self._plot_results()

    def _plot_results(self):
        df = pd.read_csv(self.results_file)
        _, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(df['epoch'], df['train/loss'], label='train')
        ax[0].plot(df['epoch'], df['val/loss'], label='val')
        ax[0].set_title('Loss'); ax[0].legend()
        ax[1].plot(df['epoch'], df['train/accuracy'], label='train_acc')
        ax[1].plot(df['epoch'], df['val/accuracy'], label='val_acc')
        ax[1].set_title('Accuracy'); ax[1].legend()
        plt.savefig(self.save_dir / "results.png", dpi=300)
        plt.close()

    def save_final_metrics(self, y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = confusion_matrix(y_true, y_pred, normalize='true')

        for matrix, name in zip([cm, cm_norm], ["confusion_matrix.png", "confusion_matrix_normalized.png"]):
            if 'norm' in name:
                annot = [[f"{v:.2f}" if v != 0 else "" for v in row] for row in matrix]
            else:
                annot = [[f"{int(v)}" if v != 0 else "" for v in row] for row in matrix]

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                matrix,
                annot=annot,
                fmt="",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names
            )

            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(self.save_dir / name, dpi=300)
            plt.close()