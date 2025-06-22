import cv2
from PIL import Image
from time import time
import numpy as np
import onnxruntime as ort
from utils import decode_function, BeamDecoder


print(ort.__version__)

model_path = "stn_lpr_opt.onnx"

CHARS = [
     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
     'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T',
     'Y', 'X', '-'
]


since = time()
image = Image.open("./val/img/Y958PX102.png")


mean = np.array([0.496, 0.502, 0.504], dtype=np.float32)
std = np.array([0.254, 0.2552, 0.2508], dtype=np.float32)

def batch_transform(rgb_batch, mean, std):
    """
    Обрабатывает батч изображений: resize, transpose, normalize.

    Параметры:
        rgb_batch (np.ndarray): Входной батч изображений в формате (B, H, W, C)
        mean (list or np.ndarray): Средние значения для каждого канала (C,)
        std (list or np.ndarray): Стандартные отклонения для каждого канала (C,)

    Возвращает:
        np.ndarray: Нормализованный батч в формате (B, C, H, W)
    """
    B, H, W, C = rgb_batch.shape

    # Конвертируем в float32 и делим на 255, если это uint8
    if rgb_batch.dtype == np.uint8:
        rgb_batch = rgb_batch.astype(np.float32) / 255.0

    # Resize (24, 94) — высота 24, ширина 94
    resized_batch = np.zeros((B, 24, 94, C), dtype=np.float32)
    for i in range(B):
        resized_batch[i] = cv2.resize(rgb_batch[i], (94, 24))  # (W, H)

    # Перевод в формат (B, C, H, W)
    transposed = np.transpose(resized_batch, (0, 3, 1, 2))  # (B, C, H, W)

    # Нормализация по каналам
    for i in range(transposed.shape[1]):  # для каждого канала
        transposed[:, i, :, :] = (transposed[:, i, :, :] - mean[i]) / std[i]

    contiguous_transposed = np.ascontiguousarray(transposed)

    return contiguous_transposed.astype(np.float32)


sess_options = ort.SessionOptions()


session = ort.InferenceSession(
    model_path, providers=['CPUExecutionProvider'], sess_options=sess_options
)  
model_inputs = session.get_inputs()

# batch = np.stack([transform(image) for i in range(8)])

mean = [0.496, 0.502, 0.504]
std = [0.254, 0.2552, 0.2508]

times = []

BATCH_SIZE = 10

images = np.stack([image.copy() for i in range(BATCH_SIZE)])

for i in range(50):
    start = time()
    data = batch_transform(images, mean, std)

    predictions = session.run(None, {"input": data})
    labels, prob, pred_labels = decode_function(predictions[0], CHARS, BeamDecoder)
    times.append(time() - start)

print(f"{sum(times) * 1000 / len(times) / BATCH_SIZE:.3f} ms")
