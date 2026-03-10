import numpy as np
import cv2
from PIL import Image
import io
import os
from tqdm import tqdm

class DistortionGenerator:
    def __init__(self):
        self.img_uint8 = None
        self.img = None
        
        self.height = None
        self.width = None
        self.channels = None
        
        self.img_name = None
        self.folder_path = None
        self.root_path = None
    
    def change_image(self, new_image_path: str):
        """Permite alterar a imagem de entrada."""
        self.img_uint8 = cv2.imread(new_image_path)

        if self.img_uint8 is None:
            raise ValueError(f"Image not found at path: {new_image_path}.")
        
        # Converte para float32 no intervalo [0, 1]
        self.img = self.img_uint8.astype(np.float32) / 255.0
        self.height, self.width, self.channels = self.img.shape
        self.img_name = self._get_image_name(new_image_path)
        self.folder_path = os.path.dirname(new_image_path)
        self.root_path = os.path.dirname(self.folder_path)

    def _get_image_name(self, image_path: str):
        return os.path.splitext(os.path.basename(image_path))[0]
    
    def _check_image_loaded(self):
        if self.img_uint8 is None:
            raise ValueError("No image loaded. Use change_image() to load an image.")

    def save_output(self, distorted_img, distortion_name, level):
        """Saves the resulting image converting back to uint8."""
        self._check_image_loaded()
        # Garante que os valores fiquem entre 0 e 1 antes de converter
        img_clipped = np.clip(distorted_img, 0, 1)
        img_out = (img_clipped * 255).astype(np.uint8)
        save_path = f'{self.root_path}/{distortion_name}'
        os.makedirs(save_path, exist_ok=True)
        file_path = f"{save_path}/dist_{level}_{self.img_name}.png"
        cv2.imwrite(file_path, img_out, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    # 1. Gaussian Blur (Kernel e Sigma dinâmicos)
    def add_gaussian_blur(self, sigma=2.0, force_kernel=None):
        """
        Applies a Gaussian filter with dynamic kernels.
        Se force_kernel for None, o OpenCV calcula o kernel ideal baseado no sigma.
        """
        self._check_image_loaded()
        
        if force_kernel is not None:
            # Garante que o kernel seja ímpar
            k = int(force_kernel)
            k = k if k % 2 != 0 else k + 1
            kernel_size = (k, k)
        else:
            # (0,0) faz o cv2 calcular o tamanho do kernel automaticamente usando a fórmula do sigma
            kernel_size = (0, 0)
            
        blurred = cv2.GaussianBlur(self.img, kernel_size, sigma)
        return blurred

    # 2. Poisson-Gaussian Noise (Signal-Dependent Noise)
    def add_poisson_gaussian_noise(self, shot_noise=0.01, read_noise=0.001):
        """
        Simula ruído realista de sensor.
        shot_noise: ruído dependente dos fótons (afeta mais as áreas claras)
        read_noise: ruído eletrônico base (afeta a imagem toda, visível nas sombras)
        """
        self._check_image_loaded()
        
        # Variância total = (intensidade * shot_noise) + read_noise
        # Criamos um mapa de desvio padrão (sigma) para cada pixel
        variance_map = (self.img * shot_noise) + read_noise
        sigma_map = np.sqrt(np.clip(variance_map, 0, None))
        
        # Gera ruído normal padronizado e multiplica pelo mapa de sigma
        noise = np.random.normal(0, 1, self.img.shape) * sigma_map
        
        noisy_img = self.img + noise
        return noisy_img

    # 3. JPEG Compression
    def add_jpeg_compression(self, quality=10):
        self._check_image_loaded()
        pil_img = Image.fromarray(np.clip(self.img * 255, 0, 255).astype(np.uint8))
        buffer = io.BytesIO()
        pil_img.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        jpeg_img = Image.open(buffer)
        img_np = np.array(jpeg_img)
        return img_np.astype(np.float32) / 255.0

    # 4. JPEG-2000 Compression
    def add_jpeg2000_compression(self, compression_ratio=20):
        self._check_image_loaded()
        img_u8 = np.clip(self.img * 255, 0, 255).astype(np.uint8)
        temp_filename = f"temp_dist_{np.random.randint(1000,9999)}.jp2" # Evita conflito de arquivos
        try:
            cv2.imwrite(temp_filename, img_u8, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, compression_ratio * 10])
            jp2_img = cv2.imread(temp_filename)
            os.remove(temp_filename)
            if jp2_img is not None:
                return jp2_img.astype(np.float32) / 255.0
            return self.img
        except Exception as e:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return self.img

    # 5. Global Contrast Decrements
    def change_contrast(self, alpha=0.5):
        self._check_image_loaded()
        gray_mean = 0.5
        contrast_img = alpha * (self.img - gray_mean) + gray_mean
        return contrast_img

    # 6. Additive Pink Gaussian Noise (com variação espacial leve)
    def add_pink_noise(self, intensity=0.05, spatial_scale=1.0):
        self._check_image_loaded()
        def pink_noise_2d(shape):
            rows, cols = shape
            u = np.fft.fftfreq(rows)
            v = np.fft.fftfreq(cols)
            u_grid, v_grid = np.meshgrid(u, v, indexing='ij')
            
            f = np.sqrt(u_grid**2 + v_grid**2)
            f[0, 0] = 1.0 
            
            # spatial_scale permite alterar a "textura" do ruído rosa
            scale = 1.0 / (f**spatial_scale + 1e-5)
            
            white_noise_spec = np.fft.fft2(np.random.normal(0, 1, shape))
            pink_spec = white_noise_spec * scale
            pink_noise = np.fft.ifft2(pink_spec).real
            
            pink_noise = (pink_noise - np.mean(pink_noise)) / (np.std(pink_noise) + 1e-8)
            return pink_noise

        noise_layer = np.zeros_like(self.img)
        for ch in range(self.channels):
            noise_layer[:,:,ch] = pink_noise_2d((self.height, self.width))
            
        noisy_img = self.img + (noise_layer * intensity)
        return noisy_img


if __name__ == "__main__":
    try:
        generator = DistortionGenerator() 

        folder_path = '/home/jmn/Dev/Datasets/Distortions_v3/test/src'
        if not os.path.exists(folder_path):
            print(f"Diretório não encontrado: {folder_path}")
            files = []
        else:
            files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        for file in tqdm(files):
            image_path = os.path.join(folder_path, file)
            try:
                generator.change_image(image_path)
            except Exception as e:
                print(f"Erro ao carregar imagem {file}: {e}")
                continue
            
            # 1. Blur Dinâmico
            sigma_value = np.random.uniform(0.8, 4.0)
            # 50% de chance de deixar o cv2 calcular o kernel ideal, 50% de chance de forçar um kernel estranho (ex: kernel pequeno com sigma alto)
            k_force = np.random.choice([None, np.random.randint(3, 15)]) 
            res_blur = generator.add_gaussian_blur(sigma=sigma_value, force_kernel=k_force)
            generator.save_output(res_blur, "blur", f's{sigma_value:.2f}_k{k_force}')

            # 2. Poisson-Gaussian Noise (Ruído Realista)
            # shot_noise domina altas luzes, read_noise domina baixas luzes
            shot_v = np.random.uniform(0.005, 0.05)
            read_v = np.random.uniform(0.0001, 0.005)
            res_pgn = generator.add_poisson_gaussian_noise(shot_noise=shot_v, read_noise=read_v)
            generator.save_output(res_pgn, "awgn", f'sh{shot_v:.3f}_rd{read_v:.4f}')

            # 3. JPEG
            quality_value = np.random.randint(5, 30)
            res_jpg = generator.add_jpeg_compression(quality=quality_value)
            generator.save_output(res_jpg, "jpeg", quality_value)

            # 4. JPEG 2000
            compression_ratio = np.random.randint(1, 20)
            res_jp2 = generator.add_jpeg2000_compression(compression_ratio=compression_ratio)
            generator.save_output(res_jp2, "jpeg2000", compression_ratio)

            # 5. Contrast Decrement
            alpha_value = np.random.uniform(0.08, 0.6)
            res_contrast = generator.change_contrast(alpha=alpha_value)
            generator.save_output(res_contrast, "contrast", f'{alpha_value:.2f}')
            
            # 6. Pink Noise (Variação na escala espacial)
            intensity_value = np.random.uniform(0.03, 0.2)
            spatial_scale_val = np.random.uniform(0.8, 1.5) # Altera a frequência do ruído rosa
            res_pink = generator.add_pink_noise(intensity=intensity_value, spatial_scale=spatial_scale_val)
            generator.save_output(res_pink, "fnoise", f'i{intensity_value:.2f}_s{spatial_scale_val:.2f}')

    except Exception as e:
        print(f"Erro fatal: {e}")