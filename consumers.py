import os
from collections import deque
from queue import Queue
from threading import Lock
from time import sleep, time


import cv2
import numpy as np
from model_onnx import YoloONNX
from PyQt6.QtCore import QThread, pyqtSignal





class VideoThread(QThread):
    """
    class for handling videosource
    """
    position_signal = pyqtSignal(int)

    def __init__(self, output_queue: Queue, video_filename: str, *args, **kwargs):
        """
        video_filename - path to video
        output_queue - queue for storing frames 

        """
        super().__init__(*args, **kwargs)
        self._run_flag = True
        self.source = video_filename
        self.output_queue = output_queue
        self.stream_lock = Lock()
        self.cap = None #opencv capture
        self.fps = 0 
        self.total_frames = 0 #total frames of videofile

    def run(self):
        """
        implement thread task
        """
        self.cap = cv2.VideoCapture(self.source)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        delay = 1.0 / self.fps #fetch frames from file at fps

        # preallocate arrays
        cv_readed = np.zeros(
            shape=(cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH, 3),
            dtype=np.uint8,
        )
        rgb_frame = np.zeros(
            shape=(cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH, 3),
            dtype=np.uint8,
        )

        frame_cnt = 0
        while self._run_flag:
            with self.stream_lock:
                ret, frame = self.cap.read(cv_readed)
                if ret:
                    self.output_queue.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, rgb_frame))

            frame_cnt += 1

            if frame_cnt % (self.fps) == 0:
                pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.position_signal.emit(int(pos / self.total_frames * 1000))
            sleep(delay)

        self.cap.release()

    def stop(self):
        self._run_flag = False

    def restart(self):
        self._run_flag = True

    def set_position(self, new_pos: int):
        """
        update position in video file
        """
        if self.cap is not None and self.cap.isOpened():
            with self.stream_lock:
                self.cap.set(
                    cv2.CAP_PROP_POS_FRAMES, (new_pos / 1000 * self.total_frames)
                )

    @property
    def duration(self):
        """
        return total duration of video in seconds
        """
        return self.total_frames // self.fps


class OnnxThread(QThread):

    def __init__(
        self,
        input_queue: Queue,
        output_queue: Queue,
        onnx_model: str = None,
        device_mode="cpu",
        confidence=0.5,
        *args,
        **kwargs,
    ):
        """
        input_queue - frame queue for processing
        output_queue - output queue for storing frames 
        """
        super().__init__(*args, **kwargs)

        if not onnx_model:
            raise ValueError("Provide parameter onnx_model as model path")

        self._run_flag = True
        self.input_queue = input_queue
        self.output_queue = output_queue
        if device_mode == "cpu":
            self.batch = max(
                4, os.cpu_count() - 4
            )  # остальная система тоже хочет ресурсов
        else:
            self.batch = 8  # 8images batch for gpu
        self.model = YoloONNX(
            onnx_model, device=device_mode, batch=self.batch, confidence=confidence, labels=['Top']
        )

        self.fps_deq = deque(maxlen=10) #mean of last 10 fpr measure  
    @property
    def fps(self):
        """
        onnx model performance fps
        """
        fps = (
            "0"
            if not len(self.fps_deq)
            else f"{1 / (sum(self.fps_deq) / len(self.fps_deq)):.2f}"
        )
        return fps

    def run(self):

        while self._run_flag:
            if self.input_queue.qsize() >= self.batch:

                start = time()
                frames = self.model(self.input_queue.batch(self.batch))
                self.fps_deq.append((time() - start) / self.batch)

                for frame in frames:
                    self.output_queue.put_nowait(frame)
            elif self.input_queue.qsize() == 0:
                sleep(0.1)
            else:
                sleep(0.02)

    def stop(self):
        self._run_flag = False

    def restart(self):
        self._run_flag = True
