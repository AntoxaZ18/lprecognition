from inference_engine import Inference, ModelLoadFS
from PIL import Image
import numpy as np
from lprnet_postprocess import lprnet_decode
from preprocess import YoloPostProcess, crop_image
from time import sleep, time
from concurrent.futures import as_completed
from utils import decode_function, BeamDecoder
import cv2
from queue import Queue

from fastuuid import uuid4
from threading import Thread
from typing import Tuple


Model_shape = (320, 320)

inf = Inference(ModelLoadFS('./models'))
inf.create_session('ex6_c3k2_light_.onnx', "yolo_lp", args = {"model_shape": Model_shape, "bgr":True})
inf.create_session('stn_lpr_opt_2.onnx', "lpr_recognition", args = {"model_shape":(24, 94)})


CHARS = [
     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
     'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T',
     'Y', 'X', '-'
]

class VideoPipeLine:
    def __init__(self, video_source: str, yolo_shape=(640, 640)):
        self.source = video_source
        self._stop = False
        self.src_queue = Queue()
        self.output_queue = Queue()
        self.yolo_shape = yolo_shape

        self.src = Thread(target=self.read_thread, daemon=True)
        self.process = Thread(target=self.pipeline, daemon=True)
        self.video = Thread(target=self.show_video, daemon=True)
        self.fps = 30
        self.weight = None
        self.height = None

    def read_thread(self):
        cap = cv2.VideoCapture(self.source)
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


        # print(self.fps, total_frames)

        # img = cv2.imread("LPR.png")
        # print(img.shape)

        # self.height = img.shape[0]
        # self.width = img.shape[1]
        
        # preallocate arrays
        cv_readed = np.zeros(
            shape=(self.height, self.width, 3),
            dtype=np.uint8,
        )

        while not self._stop:
            ret, frame = cap.read(cv_readed)
            if ret:
                self.src_queue.put(frame)
            # frame = cv2.imread("vesta.png")

            self.src_queue.put(frame)
            sleep(5 / self.fps)

        # cap.release()


    def start(self):
        self.src.start()
        self.process.start()
        self.video.start()

    def show_video(self):
                    
        count = 0            
        while True:
            # try:
            #     image = self.output_queue.get(timeout=0.5)
            #     cv2.imwrite(f"./frames/{count}.png", image)
            #     count += 1
            # except Exception as e:
            #     print(e)


            try:
                cv2.imshow('Video Player', self.output_queue.get_nowait())

                if cv2.waitKey(25) & 0xFF == ord('q'): # Press 'q' to quit
                    break
            except Exception as e:
                continue

            sleep(1 / self.fps)



    def kill(self):
        self._stop = True


    def draw_box(self, image, coords, model_shape: Tuple) -> None:
        import sys

        ratio = model_shape[1] / self.width
        pad = (model_shape[0] - int(self.height * ratio)) / 2 

        for obj_coord in coords:
            obj_coord = np.array(obj_coord).astype(float)

            obj_coord /= ratio
            obj_coord[1] -= pad / ratio

            obj_coord = obj_coord.astype(int)

            box = (obj_coord[0], obj_coord[1], obj_coord[2], obj_coord[3])

            cv2.rectangle(image, box, (255, 0, 0), 2)

    def pipeline(self):
        yolo_postprocess = YoloPostProcess()
        yolo_batch_results = list()

        image_batch = []
        lpr_images = []

        while not self._stop:

            while not self.src_queue.empty():
                image_batch.append(self.src_queue.get_nowait())

            if not image_batch:
                sleep(0.1)
                continue


            start = time()
            futures = [inf.submit_task(model_id = "yolo_lp", model_inputs = image, task_id = uuid4()) for image  in image_batch]

            inference_batch_size = len(futures)


            for img, fut in zip(image_batch, futures):
                outputs = fut.result()
                results = yolo_postprocess(outputs)

                if not results:
                    continue

                tiles = crop_image(img, results['boxes'], model_shape=self.yolo_shape)
                lpr_images += tiles
                yolo_batch_results.append(results)
                # cv2.imwrite('crop.png', tiles[0])


            futures = [inf.submit_task(model_id = "lpr_recognition", model_inputs = image, task_id = uuid4()) for image  in lpr_images]

            for fut in as_completed(futures):
                result = fut.result()
            #     name = img.split('\\')[-1].split('.')[0]
                predicted = lprnet_decode(np.expand_dims(result, 0))[0][0]
                print(predicted)

            ms = (time() - start) * 1000
            # print(f"total: {ms:.2f}, per image: {ms / inference_batch_size:.2f} ms {inference_batch_size}")

            for image, result in zip(image_batch, yolo_batch_results):
                self.draw_box(image, result['boxes'], model_shape=self.yolo_shape)
                self.output_queue.put(image.copy())

            image_batch.clear()
            lpr_images.clear()
            yolo_batch_results.clear()


# import os
# print(os.cpu_count())

# threads_num = 6

# threads = [Thread(target=pipeline, daemon=True) for _ in range(threads_num)]

# for t in threads:
#     t.start()
#     sleep(0.01)

video_stream = VideoPipeLine("video.mp4", yolo_shape=Model_shape)

video_stream.start()

while True:
    sleep(1)