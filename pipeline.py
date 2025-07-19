from collections import deque
from concurrent.futures import as_completed
from queue import Queue
from threading import Thread, get_ident
from time import sleep, time
from typing import Tuple

import cv2
import numpy as np

from inference_engine import Inference
from lprnet_postprocess import lprnet_decode
from preprocess import PlateFilter, YoloPostProcess, crop_image
from tracker import ByteTracker
from utils import CHARS, IdGen


class ResultFilter:
    def __init__(self, length=10):
        self.max_size = length
        self._queue = deque(maxlen=length)
        self._set = set()

    def add(self, item):
        """
        return item if result is new None if result exists
        """
        if item in self._set:
            return
        if len(self._set) >= self.max_size:
            oldest = self._queue.popleft()
            self._set.remove(oldest)
        self._set.add(item)
        self._queue.append(item)

        return item

    def __contains__(self, item):
        return item in self._set


class VideoPipeLine:
    def __init__(
        self,
        video_source: str,
        inference: Inference,
        config: dict,
        output_queue=None,
        result_queue: Queue = None,
        rgb_out=True,
    ):
        self.source = video_source
        self._stop = False
        self.src_queue = Queue()
        self.sink = output_queue

        self.src = Thread(target=self.read_thread, daemon=True)
        self.process = Thread(target=self.pipeline, daemon=True)
        self.video = Thread(target=self.show_video, daemon=True)
        self.fps = 30
        self.weight = None
        self.height = None
        self.inference = inference
        self.rgb_out = rgb_out
        self.result_queue = result_queue
        self.result_filter = ResultFilter()

        self.stages = {}

        for stage, stage_cfg in config.items():
            name = stage_cfg["name"]
            model = stage_cfg["model"]
            args = stage_cfg.get("args", {})
            self.stages[stage] = name

            self.inference.create_session(model, name, args=args)
            self.inference.create_session(model, name, args=args)

        if "320" in config["1"]["model"]:
            self.detect_model_shape = (320, 320)
        elif "640" in config["1"]["model"]:
            self.detect_model_shape = (640, 640)

        print(self.stages)

    def read_thread(self):
        cap = cv2.VideoCapture(self.source)
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))

        print(self.fps)

        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # preallocate arrays
        cv_readed = np.zeros(
            shape=(self.height, self.width, 3),
            dtype=np.uint8,
        )

        while not self._stop:
            ret, frame = cap.read(cv_readed)
            if ret:
                self.src_queue.put(frame)
                # self.sink.put(frame)

                sleep(2 / self.fps)

    def start(self):
        self.src.start()
        self.process.start()
        # self.video.start()

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
                cv2.imshow("Test", self.sink.get_nowait())

                if cv2.waitKey(25) & 0xFF == ord("q"):  # Press 'q' to quit
                    break
            except Exception as e:
                continue

            sleep(1 / self.fps)

    def stop(self):
        self._stop = True

    def draw_box(
        self, image, box_coords: list[int], model_shape: Tuple, desc: str
    ) -> None:
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
            cv2.putText(
                image, desc, (box[0], box[1] - 10), font, fontScale, color, thickness
            )

    def run_inference(
        self, model_id: str, image_batch: dict[str, np.ndarray], postprocess=None
    ) -> dict[str, dict]:
        results = {}
        futures = [
            self.inference.submit_task(
                model_id=model_id, model_inputs=image, task_id=uuid
            )
            for uuid, image in image_batch.items()
        ]

        # collect results
        for future in as_completed(futures):
            predicts, image_id = future.result()
            if postprocess:
                results[image_id] = postprocess(predicts)
            else:
                results[image_id] = predicts

        return results

    def resample_images(self, image_batch: dict, batch_size: int):
        keys = list(image_batch.keys())

        step = len(image_batch) / batch_size
        resampled = {}
        for i in range(batch_size):
            idx = keys[int(i * step)]
            resampled[idx] = image_batch[idx]

        return resampled

    def pipeline(self):
        yolo_postprocess = YoloPostProcess(
            confidence=0.3,
            iou=0.15,
            tracker=ByteTracker(
                frame_rate=15,
                track_high_thresh=0.5,  # Порог уверенности для начала трека
                track_low_thresh=0.2,
                track_buffer=150,
                match_thresh=0.9,
                new_track_thresh=0.5,
                fuse_score=0.4,
            ),
        )

        def lpr_postprocess(prediction):
            return lprnet_decode(np.expand_dims(prediction, 0))[0][0]

        image_batch = dict()
        lpr_images = dict()
        ocr_filters = dict()

        uuid_gen = IdGen(thread_id=get_ident())

        batch_len = 8

        lpr_track = {}

        while not self._stop:
            while not self.src_queue.empty():
                image_batch[uuid_gen()] = self.src_queue.get_nowait()

            if not image_batch:
                sleep(0.01)
                continue

            if len(image_batch) > batch_len:
                new_len = max(batch_len - (len(image_batch) - batch_len) // 2, 4)
                image_batch = self.resample_images(image_batch, new_len)

            start = time()

            yolo_predicts = self.run_inference(
                self.stages["1"], image_batch, yolo_postprocess
            )

            # print("yolo_predicts", yolo_predicts)

            image_batch = dict(
                sorted(image_batch.items())
            )  # sort dict of images by uuid / timestamp

            for uuid, img in image_batch.items():
                result = yolo_predicts[uuid]

                if not result:
                    continue

                for detection in result:
                    object_uuid = uuid_gen()

                    track = detection.get("track_id")

                    if track:
                        ocr_filters[track] = ocr_filters.get(track, PlateFilter())

                        box = crop_image(
                            img, detection["box"], model_shape=self.detect_model_shape
                        )

                        lpr_images[object_uuid] = box
                        lpr_track[object_uuid] = detection  # connect lpr with track
                        # cv2.imwrite('crop.png', box)

                # print(len(result))

                # print(len(results["boxes"]), len(results["track_id"]), len(results["track_id"]))

            # print("lpr_images", lpr_images.keys())
            # print("lpr_track", lpr_track)

            lpr_predicts = self.run_inference(
                self.stages["2"], lpr_images, lpr_postprocess
            )

            for object_uuid, decoded in lpr_predicts.items():
                # print(object_uuid, decoded)
                obj_detection = lpr_track[object_uuid]

                if obj_detection["track_id"] is None:
                    continue

                score = obj_detection["score"]
                track_id = obj_detection["track_id"]

                ocr_filters[track_id].add((decoded, score))

            current_tracks = set(
                [
                    track.track_id
                    for track in yolo_postprocess.tracker.tracker.tracked_stracks
                    + yolo_postprocess.tracker.tracker.lost_stracks
                ]
            )
            ocr_filters = {
                track: ocr_filter
                for track, ocr_filter in ocr_filters.items()
                if track in current_tracks
            }

            for uuid, img in image_batch.items():
                predictions = yolo_predicts[uuid]

                for prediction in predictions:
                    track_id = prediction.get("track_id")

                    if track_id:
                        try:
                            description = (
                                f"{track_id} {ocr_filters[track_id].most_frequent()}"
                            )
                        except KeyError:
                            print("key error {track_id}")
                            continue
                    else:
                        description = ""

                    self.draw_box(
                        img,
                        prediction["box"],
                        model_shape=self.detect_model_shape,
                        desc=description,
                    )

                    if (
                        track_id
                        and ocr_filters[track_id].most_frequent()
                        and self.result_queue
                    ):
                        filtered_result = self.result_filter.add(track_id)
                        if filtered_result:
                            self.result_queue.put(ocr_filters[track_id].most_frequent())

                    # print(str(threading.get_ident()), ocr_filters[track_id].most_frequent())

                if self.rgb_out:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                self.sink.put(img)

            image_batch.clear()
            lpr_images.clear()

            sleep(0.01)

            # print(f"{(time() - start) * 1000 / 8:.3f} ms")
