from typing import Any
from abc import ABC, abstractmethod
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple
from tracker import TrackerInput, ByteTracker
from collections import Counter
import heapq
import re

@dataclass
class ModelSession:
    model_id: str
    preprocessor: 'ModelPreprocessor'
    session: Any  # Например, onnxruntime.InferenceSession

class ModelPreprocessor(ABC):
    def __init__(self, model_shape: tuple):
        """
        model_shape в формате (Height, Weight)
        """
        self.model_shape = model_shape

    @abstractmethod
    def preprocess_batch(self, rgb_batch: list[np.ndarray]) -> np.ndarray:
        ...

class LPRnetPreprocessor(ModelPreprocessor):
    """
    preprocessing images for lprnet
    """

    def __init__(self, model_shape):
        super().__init__(model_shape)

        self.mean = np.array([0.496, 0.502, 0.504], dtype=np.float32)
        self.std = np.array([0.254, 0.2552, 0.2508], dtype=np.float32)

    def preprocess_batch(self, rgb_batch: list[np.ndarray]) -> np.ndarray:
        """
        Обрабатывает батч изображений (список или кортеж) в формате RGB: resize, transpose, normalize.

        Параметры:
            rgb_batch (list, tuple) : Входной батч изображений в формате (H, W, C)

        Возвращает:
            np.ndarray: Нормализованный батч в формате (B, C, H, W) выделенный в единой памяти
        """

        rgb_batch = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in rgb_batch]

        resized_batch = [cv2.resize(img, (self.model_shape[0], self.model_shape[1])) for img in rgb_batch]

        resized_batch = np.ascontiguousarray(np.stack(resized_batch))

        # Конвертируем в float32 и делим на 255, если это uint8
        resized_batch = resized_batch.astype(np.float32) / 255.0

        # Перевод в формат (B, C, H, W)
        transposed = np.transpose(resized_batch, (0, 3, 1, 2))  # (B, C, H, W)

        transposed = (transposed - self.mean.reshape(1, -1, 1, 1)) / self.std.reshape(1, -1, 1, 1)

        return transposed


class YoloPreprocess(ModelPreprocessor):
    """
    preprocessing for yolo model
    """
    def __init__(self, model_shape, bgr=False, mode='letterbox'):
        """
        mode = crop - crop center
        mode = letterbox - letterbox image
        """
        super().__init__(model_shape)
        self.mode = mode
        self.bgr = bgr

    def preprocess_batch(self, rgb_batch: list[np.ndarray], crop=None) -> list[np.ndarray]:
        
        #конвертируем в RGB если изначально был формат opencv
        if self.bgr:
            rgb_batch = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in rgb_batch]

        rgb_batch = [self._letterbox(img) for img in rgb_batch]
        # Image.fromarray(rgb_batch[0]).save("preprocessed.jpg")

        resized_batch = np.ascontiguousarray(np.stack(rgb_batch))
        resized_batch = resized_batch.astype(np.float32) / 255.0
        resized_batch = np.transpose(resized_batch, (0, 3, 1, 2))

        return resized_batch


    def _letterbox(self, image: np.ndarray) -> np.ndarray:
        shape = image.shape[:2]
        ratio = min(self.model_shape[0] / shape[0], self.model_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
        dw, dh = (
            (self.model_shape[1] - new_unpad[0]) / 2,
            (self.model_shape[0] - new_unpad[1]) / 2,
        )  

        img = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return img

class ModelPreprocessorFactory:
    """
    class for creation preprocessors
    """
    def __init__(self):
        self._creators = {}

    def register(self, model_id:str, creator: str) -> None:
        """
        register preprocessing for model class 
        """
        self._creators[model_id] = creator

    def create(self, model_id: str, *args, **kwargs) -> ModelPreprocessor:
        """
        create preprocessor based on model type
        """
        if model_id not in self._creators:
            raise ValueError(f"Unknown model {model_id}")
        return self._creators[model_id](*args, **kwargs)
    

class YoloPostProcess:
    def __init__(self, confidence=0.5, iou=0.5, mode="letterbox", tracker=None):
        self.tracker = tracker
        self.confidence = confidence
        self.mode = mode
        self.iou = iou
        self.img_height = 320
        self.img_width = 320

    def __call__(self, model_outputs:np.ndarray):
        outputs = np.transpose(model_outputs)

        # find max score for prediction
        max_scores = np.amax(outputs[:, 4:], axis=1)
        # filter prediction with result more than confidence thresh
        filtered = outputs[max_scores > self.confidence]
        
        if len(filtered) == 0:
            return []

        # find class ids and scores
        class_ids = np.argmax(filtered[:, 4:], axis=1).tolist()
        scores = np.amax(filtered[:, 4:], axis=1).tolist()

        # convert coordinates of boxes
        x = filtered[:, 0]
        y = filtered[:, 1]
        w = filtered[:, 2]
        h = filtered[:, 3]

        left = (x - w / 2).astype(int)
        top = (y - h / 2).astype(int)
        width = w.astype(int)
        height = h.astype(int)

        boxes = np.column_stack((left, top, width, height)).tolist()

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, self.iou)

        if len(indices) == 0:
            return []

        # Выбираем только те, что прошли NMS
        boxes = [boxes[i] for i in indices]
        scores = [scores[i] for i in indices]
        class_ids = [class_ids[i] for i in indices]

        result = []

        if self.tracker:
            
            x = TrackerInput(conf=np.array(scores), xywh=np.array(boxes), cls_=np.array(class_ids))
            online_targets = self.tracker.update(x, (self.img_height, self.img_width))

            if len(online_targets):
                x1 = online_targets[:, 0]
                y1 = online_targets[:, 1]
                x2 = online_targets[:, 2]
                y2 = online_targets[:, 3]

                boxes = np.column_stack([(x1+x2) / 2, (y1 + y2) / 2, x2-x1, y2-y1]).tolist()
                scores =  online_targets[:, 5].tolist()
                class_ids = online_targets[:, 6].tolist()
                track_ids = online_targets[:, 4].tolist()

                for box, score, class_id, track_id in zip(boxes, scores, class_ids, track_ids):
                    result.append({
                        "box": box,
                        "score": score,
                        "class_id": class_id,
                        "track_id": track_id
                    })
        else:
            for box, score, class_id in zip(boxes, scores, class_ids):
                result.append({
                    "box": box,
                    "score": score,
                    "class_id": class_id
                })
        return result

def crop_image(image:np.ndarray, box: list[int], model_shape: Tuple[int, int], crop=None) -> np.ndarray:
    
    h, w, *_ = image.shape

    if crop:
        ratio = model_shape[0] / (crop[2] - crop[0]) 

        print(ratio)

        y1 = crop[0] 
        x1 = crop[1]
        y2 = crop[2]
        x2 = crop[3]

        pad = x1

        box = np.array(box)
        box /= ratio
        box[0] += pad
        box = box.astype(int)

        cropped = image[box[1] : box[1] + box[3], box[0] : box[0] + box[2], :]

        return cropped

    else:
        #letterbox mode
        ratio = model_shape[1] / w
        pad = (model_shape[0] - int(h * ratio)) / 2 

        box = np.array(box)
        box = box / ratio
        box[1] = box[1] - pad / ratio

        box = box.astype(int)
        box[box < 0] = 0

        cropped = image[box[1] : box[1] + box[3], box[0] : box[0] + box[2], :]

        return cropped

def crop_debug(image:np.ndarray, box: list[int], model_shape: Tuple[int, int], crop=None) -> np.ndarray:
    
    h, w, *_ = image.shape
    
    print(h, w)

    ratio = model_shape[1] / w
    pad = (model_shape[0] - int(h * ratio)) / 2 

    box = np.array(box)
    box = box / ratio
    box[1] = box[1] - pad / ratio

    box = box.astype(int)


    cropped = image[box[1] : box[1] + box[3], box[0] : box[0] + box[2], :]

    return cropped

def crop(image: np.ndarray, box: Tuple[int, int, int, int]):
    return image[box[1] : box[3], box[0] : box[2], :]



class PlateFilter:
    def __init__(self, window_size: int = 30, thresh_count = 5):
        self.window_size = window_size
        self.thresh_count = thresh_count
        self.heap = []

    def add(self, plate: Tuple[str, float]):
        if not self._validate_plate(plate[0]):
            return
        
        heapq.heappush(self.heap, (-plate[1], plate[0]))

        if len(self.heap) > self.window_size:
            heapq.heappop(self.heap)
        
    def most_frequent(self):
        '''
        return most frequent plate from heap
        '''
        if not self.heap:
            return None
        
        counter = Counter([text for _, text in self.heap])  #get only texts

        most_common = counter.most_common(1)

        if most_common and most_common[0][1] >= self.thresh_count:
            return most_common[0][0] 

        return None


    def _validate_plate(self, license_plate: str):
        return re.fullmatch(r".\d{3}.{2}\d{2,3}", license_plate)

# if __name__ == "__main__":

# if __name__ == "__main__":
#     from PIL import Image
#     from time import time
#     import onnxruntime as ort
#     from yolo_onnnx import YoloONNX

#     model_shape = (320, 320)

#     image = np.asarray(Image.open("39.bmp"))
#     # image = np.asarray(Image.open("./test.jpg"))

#     # image = cv2.imread("test.jpg")

#     batch_size = 8

#     batch = [np.asarray(image.copy()) for i in range(batch_size)]
#     # batch = [image] * batch_size

#     # model = YoloONNX("./models/ex8_c3k2_light_320_nwd_.onnx", device='CPU', threads=batch_size, classes=['LPR'])

#     # frame_boxes = model(batch)
#     # print(frame_boxes)

#     preprocess = YoloPreprocess(model_shape=model_shape, mode="letterbox")
#     postprocess = YoloPostProcess()

#     start = time()

#     # roi = crop(image, (465, 173, 1252, 525))
#     roi = image
#     batch = [roi.copy() for _ in range(batch_size)]
#     batch = preprocess.preprocess_batch(batch) 

#     session = ort.InferenceSession("./models/model.quant.onnx", providers=["CPUExecutionProvider"])

#     model_inputs = session.get_inputs()
#     input_shape = model_inputs[0].shape

#     output_name = session.get_outputs()[0].name
#     input_name = session.get_inputs()[0].name

#     print("model_shape", model_inputs[0].shape)

#     outputs = session.run([output_name], {input_name: batch})
#     print(f"{(time() - start) * 1000:3f} ms per batch")

#     print(outputs[0][1].shape)

#     outputs = outputs[0]


#     for i in range(batch_size):
#         results = postprocess(outputs[i])

#         print(results)
#         if len(results) == 0:
#             continue
#         tiles = crop_image(roi, results['boxes'], model_shape=model_shape)
#         # tiles = crop_image(image, results['boxes'], model_shape=model_shape, crop=(0, central_pad, h, h+central_pad))

#         for idx, i in enumerate(tiles):
#             img = Image.fromarray(i)
#             img.save(f'cropped_{idx}.jpg')

#         break
