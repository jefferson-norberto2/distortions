import os
import psutil
import pynvml
import yaml


class HardwareProfiler:
    def __init__(self, device_index=0):
        self.ram_usages = []
        self.gpu_utils = []
        self.vram_usages = []
        self.power_draws = []
        
        # Initialize NVIDIA Management Library
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def sample(self):
        # 1. System RAM (GB)
        ram_gb = psutil.virtual_memory().used / (1024 ** 3)
        self.ram_usages.append(ram_gb)
        
        # 2. GPU Utilization (%)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        self.gpu_utils.append(utilization.gpu)
        
        # 3. GPU VRAM (GB)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        self.vram_usages.append(mem_info.used / (1024 ** 3))
        
        # 4. Power Draw (Watts)
        # nvmlDeviceGetPowerUsage returns milliwatts, we divide by 1000 for Watts
        power_watts = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
        self.power_draws.append(power_watts)

    def save_to_yaml(self, save_path: str, filename: str = "hardware_metrics.yaml"):
        # Calculate averages safely
        avg = lambda x: sum(x) / len(x) if x else 0.0
        
        metrics = {
            "System_RAM_Usage_GB": {
                "average": round(avg(self.ram_usages), 2),
                "peak": round(max(self.ram_usages, default=0), 2)
            },
            "GPU_Processing_Usage_Percent": {
                "average": round(avg(self.gpu_utils), 2),
                "peak": round(max(self.gpu_utils, default=0), 2)
            },
            "GPU_VRAM_Allocation_GB": {
                "average": round(avg(self.vram_usages), 2),
                "peak": round(max(self.vram_usages, default=0), 2)
            },
            "Power_Consumption_Watts": {
                "average": round(avg(self.power_draws), 2),
                "peak": round(max(self.power_draws, default=0), 2)
            }
        }
        
        file_path = os.path.join(save_path, filename)
        with open(file_path, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)
            
        # Shutdown NVML to free resources
        pynvml.nvmlShutdown()
        return metrics