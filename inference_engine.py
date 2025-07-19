import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from threading import Lock, Thread
from time import sleep, time

import numpy as np
import onnxruntime as ort

from lprnet_postprocess import lprnet_decode
from preprocess import (
    LPRnetPreprocessor,
    ModelPreprocessorFactory,
    ModelSession,
    YoloPreprocess,
)


@dataclass
class ImgTask:
    task_id: str
    model_id: str
    inputs: np.array


class ModelLoader(ABC):
    """
    Базовый класс загрузки моделей из разных источников
    """

    @abstractmethod
    def load(self, model_name): ...


class ModelLoadFS(ModelLoader):
    """
    Реализация загрузки моделей из файловой системы
    """

    def __init__(self, storage: str):
        if not os.path.exists(storage):
            raise RuntimeError(f"folder {storage} not found")

        self.storage = storage

    def load(self, model_name: str):
        """
        загрузка onnx модели по имени
        """

        model_path = os.path.join(self.storage, model_name)

        if not os.path.exists(model_path):
            raise RuntimeError(f"{model_path} not existed")

        with open(model_path, "rb") as f:
            readed = f.read()

        return readed


class BatchCounter:
    def __init__(self, window_seconds=1):
        self.batches = deque()  # хранит временные метки
        self.window_seconds = window_seconds

    def add_batch(self):
        self.batches.append(time())

    @property
    def batches_per_window(self):
        now = time()
        # Удаляем все элементы, которые старше window_seconds
        while self.batches and now - self.batches[0] > self.window_seconds:
            self.batches.popleft()
        return len(self.batches)


class BatchManager:
    def __init__(self, batch_size: int = 8, timeout: float = 0.05):
        self.batch_size = batch_size
        self.timeout_sec = timeout
        self.batches = defaultdict(list)
        self.last_time = defaultdict(float)

    def add(self, model_id: str, task: ImgTask) -> bool:
        """
        Добавляем таск для модели, возвращает True если батч готов на обработку
        """
        self.batches[model_id].append(task)
        self.last_time[model_id] = time()

        return self.is_ready(model_id)

    def __iter__(self):
        return iter(tuple(self.batches.keys()))

    def is_ready(self, model_id: str):
        return len(self.batches[model_id]) >= self.batch_size or self._has_timeout(
            model_id
        )

    def _has_timeout(self, model_id: str) -> bool:
        """
        return bool if batch has timeout
        """
        return time() - self.last_time[model_id] > self.timeout_sec

    def get(self, model_id: str) -> list:
        self.last_time[model_id] = time()
        return self.batches.pop(model_id, [])

    def clear(self):
        self.batches.clear()


class Inference:
    """
    реализует инференс моделей
    """

    def __init__(
        self, model_loader: ModelLoader, pool_workers=os.cpu_count(), batch_size=8
    ):
        self.model_loader = model_loader
        self.batch_size = batch_size
        self.providers = ["CPUExecutionProvider"]
        self.sessions = {}
        self.pool = ThreadPoolExecutor(max_workers=pool_workers)
        self.task_queue = Queue()  # сюда накидываем задания на инференс
        self.result_futures = {}
        self.fut_lock = Lock()
        self.batch_manager = BatchManager(batch_size=self.batch_size)
        self.perf_metric = BatchCounter(window_seconds=1)

        self.preproc_factory = ModelPreprocessorFactory()
        self.preproc_factory.register("lpr_recognition", lambda: LPRnetPreprocessor)
        self.preproc_factory.register("yolo_lp", lambda: YoloPreprocess)

        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    @property
    def batches_per_second(self):
        return self.perf_metric.batches_per_window

    def model_img_size(self, model_name: str):
        match = re.search(r"(\d+)x(\d+)", model_name)

        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            return (width, height)
        else:
            return None

    def create_session(self, model_name: str, model_id: str, args=None) -> None:
        """
        load model and register preprocessor
        """

        model = self.model_loader.load(model_name)

        sess_options = ort.SessionOptions()
        # sess_options.graph_optimization_level = (
        #     ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # )
        # sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # sess_options.inter_op_num_threads = 3

        model_shape = self.model_img_size(model_name)

        if model_shape:
            args["model_shape"] = model_shape

        session = ort.InferenceSession(
            model, providers=self.providers, sess_options=sess_options
        )

        preprocesor = self.preproc_factory.create(model_id)(**args)

        self.sessions[model_id] = ModelSession(
            model_id=model_id, preprocessor=preprocesor, session=session
        )

    def _get_session(self, model_id: str):
        return self.sessions.get(model_id)

    def submit_task(
        self, model_id: str = None, model_inputs: np.array = None, task_id: str = None
    ):
        if not all((model_id, task_id)):
            raise ValueError("all parameters must be provided")

        future = Future()
        with self.fut_lock:
            self.result_futures[task_id] = future

        task = ImgTask(task_id=task_id, model_id=model_id, inputs=model_inputs)
        self.task_queue.put(task)
        return future

    def _worker(self):
        """
        worker function for batching data
        """
        while True:
            while not self.task_queue.empty():
                task = self.task_queue.get_nowait()

                if self.batch_manager.add(task.model_id, task):
                    tasks = self.batch_manager.get(task.model_id)
                    self.pool.submit(self._process_batch, tasks)
                    # self._process_batch(tasks)
                    sleep(0.01)

            # Проверяем, есть ли незавершённые батчи
            for model_id in self.batch_manager:
                if self.batch_manager.is_ready(model_id):
                    tasks = self.batch_manager.get(task.model_id)
                    self.pool.submit(self._process_batch, tasks)
                    # self._process_batch(tasks)

            sleep(0.01)

    def _process_batch(self, batch_tasks: list[ImgTask]):
        if not batch_tasks:
            return

        model_id = batch_tasks[0].model_id

        session = self._get_session(model_id)
        if not session:
            raise RuntimeError(f"Model {model_id} not loaded")

        batch_inputs = [task.inputs for task in batch_tasks]

        preprocessed_batch = session.preprocessor.preprocess_batch(batch_inputs)

        # inference here
        input_name = session.session.get_inputs()[0].name
        outputs = session.session.run(None, {input_name: preprocessed_batch})

        with self.fut_lock:
            for task, output in zip(batch_tasks, outputs[0]):
                future = self.result_futures.pop(task.task_id, None)
                if future:
                    future.set_result(
                        (output, task.task_id)
                    )  # task_id for trasckig result in batch

        self.perf_metric.add_batch()  # add metric

    def _get_model_id(self, model_name: str) -> str:
        if "yolo" in model_name.lower():
            return "yolo"
        elif "lprnet" in model_name.lower():
            return "lprnet"
        else:
            raise ValueError(f"Unknown model type {model_name}")


# inf = Inference(ModelLoadFS('./models'))
# inf.create_session('ex8_c3k2_light_320_nwd_.onnx', "yolo_lp")
# inf.create_session('stn_lpr_opt.onnx', "lpr_recognition")
# inputs = np.arange(1, 10)
# futures = [inf.submit_task("stn_lpr_opt.onnx", inputs, f"xxx_{i}") for i in range(10)]
# for fut in futures:
#     result = fut.result()
# # result = futures.result()
# print('ready res')
# print(result)
# sleep(1)
