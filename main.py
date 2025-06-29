import sys
from inference_engine import Inference, ModelLoadFS
import numpy as np
from lprnet_postprocess import lprnet_decode
from preprocess import YoloPostProcess, crop_image, PlateFilter
from time import sleep, time
from concurrent.futures import as_completed
import cv2
from queue import Queue
import array

from fastuuid import uuid1
from threading import Thread
from typing import Tuple
from tracker import ByteTracker
import threading

Model_shape = (320, 320)

inf = Inference(ModelLoadFS('./models'))
inf.create_session('ex6_c3k2_light_.onnx', "yolo_lp", args = {"model_shape": Model_shape, "bgr":True})
inf.create_session('stn_lpr_opt_2.onnx', "lpr_recognition", args = {"model_shape":(24, 94)})


CHARS = [
     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
     'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T',
     'Y', 'X', '-'
]


class IdGen:
    def __init__(self, thread_id, max_count = 1024):
        self.thread_id = thread_id
        self.max_count = max_count
        self.counter = 0

    def __call__(self):
        now = int(time() * 1e6)  # Микросекунды
        uuid = f"{now}-{self.thread_id}-{self.counter}"
        self.counter += 1
        return uuid

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


    def draw_box(self, image, box_coords: list[int], model_shape: Tuple, desc: str) -> None:

        ratio = model_shape[1] / self.width
        pad = (model_shape[0] - int(self.height * ratio)) / 2 

        obj_coord = np.array(box_coords).astype(float)

        obj_coord /= ratio
        obj_coord[1] -= pad / ratio

        obj_coord = obj_coord.astype(int)

        box = (obj_coord[0], obj_coord[1], obj_coord[2], obj_coord[3])

        cv2.rectangle(image, box, (255, 0, 0), 5)
        if desc:
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(image, desc, (box[0], box[1] - 10), font, fontScale, color, thickness)


    def run_inference(self, model_id: str, image_batch: dict[str, np.ndarray], postprocess=None) -> dict[str, dict]: 
        results = {}
        futures = [inf.submit_task(model_id = model_id, model_inputs = image, task_id = uuid) for uuid, image  in image_batch.items()]

        #collect results
        for future in as_completed(futures):
            predicts, image_id = future.result()
            if postprocess:

                results[image_id] = postprocess(predicts)
            else:
                results[image_id] = predicts

        return results




    def pipeline(self):
        yolo_postprocess = YoloPostProcess(confidence=0.5, iou=0.3, tracker=ByteTracker(frame_rate=10, track_high_thresh = 0.5, track_buffer=300,  new_track_thresh=0.6, fuse_score=0.4))

        def lpr_postprocess(prediction):
            return lprnet_decode(np.expand_dims(prediction, 0))[0][0]


        image_batch = dict()
        lpr_images = dict()
        ocr_filters = dict()

        uuid_gen = IdGen(thread_id=threading.get_ident())

        while not self._stop:

            while not self.src_queue.empty():
                image_batch[uuid_gen()] = self.src_queue.get_nowait()

            if not image_batch:
                sleep(0.05)
                continue


            start = time()

            yolo_predicts = self.run_inference("yolo_lp", image_batch, yolo_postprocess)

            # print("yolo_predicts", yolo_predicts)

            image_batch = dict(sorted(image_batch.items())) #sort dict of images by uuid / timestamp

            lpr_track = {}
            for uuid, img in image_batch.items():
                result = yolo_predicts[uuid]

                if not result:
                    continue

                for detection in result:

                    object_uuid = uuid_gen()

                    track = detection.get("track_id")

                    if track:
                        ocr_filters[track] = ocr_filters.get(track, PlateFilter())

                        box = crop_image(img, detection['box'], model_shape=self.yolo_shape)

                        lpr_images[object_uuid] = box
                        lpr_track[object_uuid] = detection  #connect lpr with track


                # print(len(result))

                # print(len(results["boxes"]), len(results["track_id"]), len(results["track_id"]))
                # cv2.imwrite('crop.png', tiles[0])


            # print("lpr_images", lpr_images.keys())
            # print("lpr_track", lpr_track)
            

            lpr_predicts = self.run_inference("lpr_recognition", lpr_images, lpr_postprocess)
            
            for object_uuid, decoded in lpr_predicts.items():
                # print(object_uuid, decoded)
                obj_detection = lpr_track[object_uuid]

                if obj_detection["track_id"] is None:
                    continue

                score = obj_detection["score"]
                track_id = obj_detection["track_id"]

                ocr_filters[track_id].add((decoded, score))

            current_tracks = set([track.track_id for track in yolo_postprocess.tracker.tracker.tracked_stracks + yolo_postprocess.tracker.tracker.lost_stracks])
            ocr_filters = {track: ocr_filter for track, ocr_filter in ocr_filters.items() if track in current_tracks}


            for uuid, img in image_batch.items():   
                predictions = yolo_predicts[uuid]
                
                # print(len(predictions))
                for prediction in predictions:

                    track_id = prediction.get("track_id")

                    if track_id:
                        description = f"{track_id} {ocr_filters[track_id].most_frequent()}" 
                    else:
                        description = ""
                    self.draw_box(img, prediction['box'], model_shape=self.yolo_shape, desc=description)
                    self.output_queue.put(img)

            image_batch.clear()
            lpr_images.clear()


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