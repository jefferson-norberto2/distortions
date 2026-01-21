import numpy as np
import cv2
from PIL import Image
import io
import os

class DistortionGenerator:
    def __init__(self):
        self.img_uint8 = None
    
    def change_image(self, new_image_path: str):
        """Permite alterar a imagem de entrada."""
        self.img_uint8 = cv2.imread(new_image_path)
        if self.img_uint8 is None:
            raise ValueError(f"Image not found at path: {new_image_path}.")
        self.img = self.img_uint8.astype(np.float32) / 255.0
        self.h, self.w, self.c = self.img.shape
    
    def _check_image_loaded(self):
        if self.img_uint8 is None:
            raise ValueError("No image loaded. Use change_image() to load an image.")

    def save_output(self, distorted_img, distortion_name, level):
        """Saves the resulting image converting back to uint8."""
        # Clip to ensure range [0,1] and convert to [0,255]
        self._check_image_loaded()
        img_clipped = np.clip(distorted_img, 0, 1)
        img_out = (img_clipped * 255).astype(np.uint8)
        filename = f"dist_{distortion_name}_level_{level}.png"
        cv2.imwrite(filename, img_out)
        return img_out

    def add_gaussian_blur(self, kernel_size=(15, 15), sigma=5):
        """
        Applies a Gaussian filter.
        Sigma controls the blur intensity.
        """
        # The paper mentions "Gaussian blurring" as distortion in LIVE and CSIQ
        self._check_image_loaded()
        blurred = cv2.GaussianBlur(self.img, kernel_size, sigma)
        return blurred

    # 2. Additive White Gaussian Noise (White Noise) 
    def add_white_noise(self, variance=0.01):
        """
        Adds Gaussian white noise.
        Variance controls the noise intensity.
        """
        self._check_image_loaded()
        noise = np.random.normal(0, np.sqrt(variance), self.img.shape)
        noisy_img = self.img + noise
        return noisy_img

    # 3. JPEG Compression 
    def add_jpeg_compression(self, quality=10):
        """
        Simulates JPEG compression artifacts.
        Quality ranges from 1 (worst) to 95 (best).
        """
        self._check_image_loaded()
        pil_img = Image.fromarray((self.img * 255).astype(np.uint8))
        buffer = io.BytesIO()
        # Saves to buffer with desired quality
        pil_img.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        # Reads back
        jpeg_img = Image.open(buffer)
        # Converts back to cv2 format (BGR) and float
        img_np = np.array(jpeg_img)
        # Pillow is RGB, OpenCV is BGR if not converted, but assuming input cv2:
        # If the original input was BGR (cv2 default), Pillow read as RGB incorrectly if not converted.
        # Safe adjustment: Maintain channel consistency.
        return img_np.astype(np.float32) / 255.0

    # 4. JPEG-2000 Compression 
    def add_jpeg2000_compression(self, compression_ratio=20):
        """
        Simula compressão JPEG 2000. Requer suporte a escrita .jp2 no OpenCV.
        Alternativa: Usar imageio se o cv2 falhar.
        """
        self._check_image_loaded()
        img_u8 = (self.img * 255).astype(np.uint8)
        # Tenta salvar temporariamente como jp2
        temp_filename = "temp_dist.jp2"
        try:
            # A compressão no cv2 para JPG2000 é controlada por CV_IMWRITE_JPEG2000_COMPRESSION_X1000
            # Valor alto = menor qualidade (mais compressão)
            cv2.imwrite(temp_filename, img_u8, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, compression_ratio * 10])
            jp2_img = cv2.imread(temp_filename)
            os.remove(temp_filename)
            return jp2_img.astype(np.float32) / 255.0
        except Exception as e:
            print(f"Erro ao gerar JPEG2000 (verifique drivers OpenCV): {e}")
            return self.img

    # 5. Global Contrast Decrements 
    def change_contrast(self, alpha=0.5):
        """
        Reduz o contraste global.
        Alpha < 1.0 reduz o contraste (aproxima do cinza médio).
        """
        # Fórmula: pixel = alpha * (pixel - meio) + meio
        # Assume meio como 0.5 para imagens float
        self._check_image_loaded()
        gray_mean = 0.5
        contrast_img = alpha * (self.img - gray_mean) + gray_mean
        return contrast_img

    # 6. Additive Pink Gaussian Noise 
    def add_pink_noise(self, intensity=0.05):
        """
        Gera ruído rosa (1/f) no domínio da frequência e adiciona à imagem.
        """
        self._check_image_loaded()
        def pink_noise_2d(shape):
            rows, cols = shape
            # Cria grid de frequências
            u = np.fft.fftfreq(rows)
            v = np.fft.fftfreq(cols)
            u_grid, v_grid = np.meshgrid(u, v, indexing='ij')
            
            # Calcula frequência radial f = sqrt(u^2 + v^2)
            f = np.sqrt(u_grid**2 + v_grid**2)
            
            # Evita divisão por zero no componente DC
            f[0, 0] = 1.0 
            
            # Espectro do ruído rosa: 1 / f (para amplitude, que é sqrt(PSD))
            # Ajuste: O artigo fala "Pink Gaussian". Tipicamente isso é 1/f.
            scale = 1.0 / (f + 1e-5)
            
            # Gera ruído branco no domínio da frequência (fase aleatória)
            white_noise_spec = np.fft.fft2(np.random.normal(0, 1, shape))
            
            # Aplica o filtro rosa
            pink_spec = white_noise_spec * scale
            
            # Volta para o domínio espacial
            pink_noise = np.fft.ifft2(pink_spec).real
            
            # Normaliza para média 0 e std 1
            pink_noise = (pink_noise - np.mean(pink_noise)) / np.std(pink_noise)
            return pink_noise

        # Gera ruído rosa para cada canal separadamente
        noise_layer = np.zeros_like(self.img)
        for ch in range(self.c):
            noise_layer[:,:,ch] = pink_noise_2d((self.h, self.w))
            
        noisy_img = self.img + (noise_layer * intensity)
        return noisy_img

# --- Exemplo de Uso ---
# Substitua 'input.jpg' pela sua imagem de referência
if __name__ == "__main__":
    try:
        generator = DistortionGenerator() 

        generator.change_image('/root/Documents/dev/python/distortions/data/ECSIQ/src_imgs/both_part_2_trolley.png.png')
        
        # 1. Blur
        res_blur = generator.add_gaussian_blur(sigma=1)
        generator.save_output(res_blur, "blur", 1)

        # 2. White Noise
        res_wn = generator.add_white_noise(variance=0.01)
        generator.save_output(res_wn, "white_noise", 1)

        # 3. JPEG
        res_jpg = generator.add_jpeg_compression(quality=10) # Qualidade baixa = mais artefatos
        generator.save_output(res_jpg, "jpeg", 1)

        # 4. JPEG 2000
        res_jp2 = generator.add_jpeg2000_compression(compression_ratio=1) # Ratio alto = menor qualidade
        generator.save_output(res_jp2, "jpeg2000", 1)

        # 5. Contrast Decrement
        res_contrast = generator.change_contrast(alpha=0.5) # Valor maior reduz menos o contraste
        generator.save_output(res_contrast, "contrast", 1)
        
        # 6. Pink Noise (Específico da CSIQ)
        res_pink = generator.add_pink_noise(intensity=0.05)
        generator.save_output(res_pink, "pink_noise", 1)

        print("Distorções geradas com sucesso.")

    except Exception as e:
        print(e)