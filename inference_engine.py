import cv2
import onnxruntime as ort
import numpy as np
from abc import ABC, abstractmethod
import os
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock, Thread
from queue import Queue
from time import sleep, time
from dataclasses import dataclass
from typing import List
from collections import defaultdict

from lprnet_postprocess import lprnet_decode
from preprocess import ModelPreprocessorFactory, LPRnetPreprocessor, YoloPreprocess, ModelSession

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
    def load(self, model_name):
        ...


class ModelLoadFS(ModelLoader):
    """
    Реализация загрузки моделей из файловой системы
    """
    def __init__(self, storage: str):
        if not os.path.exists(storage):
            raise RuntimeError(f"folder {storage} not found")
        
        self.storage = storage

    def load(self, model_name:str):
        """
        загрузка onnx модели по имени
        """

        model_path = os.path.join(self.storage, model_name)

        if not os.path.exists(model_path):
            raise RuntimeError(f"{model_path} not existed")
        
        with open(model_path, 'rb') as f:
            readed = f.read()

        return readed


class BatchManager:
    def __init__(self, batch_size: int = 8, timeout: float = 0.1):
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

        if len(self.batches[model_id]) >= self.batch_size or self._has_timeout(model_id):
            return True
        return False

    def _has_timeout(self, model_id: str) -> bool:
        """
        return bool if batch has timeout
        """
        return time() - self.last_time[model_id] > self.timeout_sec

    def get(self, model_id: str) -> list:
        return self.batches.pop(model_id, [])

    def clear(self):
        self.batches.clear()



class Inference():
    """
    реализует инференс моделей
    """

    def __init__(self, model_loader: ModelLoader, pool_workers=os.cpu_count()):
        self.model_loader = model_loader
        self.providers = ['CPUExecutionProvider']
        self.sessions = {}
        self.pool = ThreadPoolExecutor(max_workers=pool_workers)
        self.task_queue = Queue()   #сюда накидываем задания на инференс
        self.result_futures = {}
        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self.fut_lock = Lock()
        self.batch_manager = BatchManager(batch_size=8)

        self.preproc_factory = ModelPreprocessorFactory()
        self.preproc_factory.register("lpr_recognition", lambda: LPRnetPreprocessor(model_shape=(24, 94)))
        self.preproc_factory.register("yolo_lp", lambda: YoloPreprocess(model_shape=(320, 320)))


    def create_session(self, model_name: str, model_id: str) -> None:
        """
        load model and register preprocessor
        """

        model = self.model_loader.load(model_name)
        session = ort.InferenceSession(model, providers=self.providers, sess_options=ort.SessionOptions())

        preprocesor = self.preproc_factory.create(model_id)

        self.sessions[model_id] = ModelSession(model_id=model_id, preprocessor=preprocesor, session=session)

    def _get_session(self, model_id: str):
        return self.sessions.get(model_id)


    def submit_task(self, model_id: str = None, model_inputs: np.array = None, task_id: str = None):
        if not all ((model_id, task_id)):
            raise ValueError("all parameters must be provided")
        
        future = Future()
        with self.fut_lock:
            self.result_futures[task_id] = future

        task = ImgTask(task_id = task_id, model_id = model_id, inputs = model_inputs)
        self.task_queue.put(task)
        return future


    def _worker(self):
        '''
        worker function for batching data
        '''
        while True:
            while not self.task_queue.empty():
                task = self.task_queue.get_nowait()

                if self.batch_manager.add(task.model_id, task):
                    self.pool.submit(self._process_batch, task.model_id)
                    # self._process_batch(task.model_id)

            sleep(0.05)

    def _process_batch(self, model_id: str):
        batch_tasks = self.batch_manager.get(model_id)
        if not batch_tasks:
            return


        session = self._get_session(model_id)
        if not session:
            raise RuntimeError(f'Model {model_id} not loaded')
        
        batch_inputs = [task.inputs for task in batch_tasks]

        preprocessed_batch = session.preprocessor.preprocess_batch(batch_inputs)

        #inference here
        input_name = session.session.get_inputs()[0].name
        outputs = session.session.run(None, {input_name: preprocessed_batch})

        with self.fut_lock:
            for task, output in zip (batch_tasks, outputs[0]):
                future = self.result_futures.pop(task.task_id, None)
                if future:
                    future.set_result(output)


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






