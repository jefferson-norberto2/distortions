import time
import random
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console

class BarProgress:
    def __init__(self, header: list, width: int):
        self.console = Console()
        self.progress = None
        self.header = header
        self.width = width
        self.colors = ['cyan', 'magenta', 'red', 'yellow', 'white', 'green', 'blue']
        
        self.__configure_header()

    def __configure_header(self):
        # 1. Cabeçalho (Header)
        header_str = ""
        for title in self.header:
            header_str += f"{title:<{self.width}}"
            
        self.console.print(header_str, style="bold underline white")

        # 2. Layout da Barra
        self.progress = Progress(
            TextColumn("{task.description}"), 
            BarColumn(bar_width=20, style="black", complete_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )
    
    def __configure_description(self, description: list):
        desc_str = ""
        for i, desc in enumerate(description):
            desc_str += f"[{self.colors[i]}]{desc:<{self.width}}[/]"
        return desc_str
    
    def start(self, total: int):
        self.progress.start()
        task_id = self.progress.add_task("", total=total)
        return task_id
    
    def update(self, task_id: int, advance: int, description: list):
        if len(description) != len(self.header):
            raise ValueError("Description length must match header length.")
        
        desc_str = self.__configure_description(description)
        self.progress.update(task_id, advance=advance, description=desc_str)
    
    def stop(self):
        self.progress.stop()

if __name__ == "__main__":
    header = ['Epoch', 'GPU_mem', 'loss', 'Instances', 'Size']
    bar = BarProgress(header=header, width=12)
    task_id = bar.start(total=300)
    
    for i in range(300):
        time.sleep(0.05)

        # Dados simulados
        loss = random.uniform(0.01, 0.02)
        gpu_val = 1.99
        epoch_str = "9/10"
        inst_val = 64
        size_val = 320

        bar.update(task_id, advance=1, description=[epoch_str, f"{gpu_val:.2f}G", f"{loss:.5f}", inst_val, size_val])
    
    bar.stop()