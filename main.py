from inference_engine import Inference, ModelLoadFS
from PIL import Image
import numpy as np
from lprnet_postprocess import lprnet_decode
from preprocess import YoloPostProcess, crop_image
from time import sleep, time
from concurrent.futures import as_completed

inf = Inference(ModelLoadFS('./models'))
inf.create_session('ex8_c3k2_light_320_nwd_.onnx', "yolo_lp")
inf.create_session('stn_lpr_opt_2.onnx', "lpr_recognition")


def pipeline():
    yolo_postprocess = YoloPostProcess()

    img = np.asarray(Image.open("39.bmp"))

    yolo_images = [img.copy() for _ in range(8)]
    
    start = time()
    futures = [inf.submit_task(model_id = "yolo_lp", model_inputs = image, task_id = f"xxx_{i}") for i, image  in enumerate(yolo_images)]

    lpr_images = []
    for img, fut in zip(yolo_images, futures):
        outputs = fut.result()
        results = yolo_postprocess(outputs)

        tiles = crop_image(img, results['boxes'], model_shape=(320, 320))
        lpr_images += tiles



    futures = [inf.submit_task(model_id = "lpr_recognition", model_inputs = image, task_id = f"xxx_{i}") for i, image  in enumerate(lpr_images)]

    for fut in as_completed(futures):
        result = fut.result()
    #     name = img.split('\\')[-1].split('.')[0]
        predicted = lprnet_decode(np.expand_dims(result, 0))[0][0]
        print(predicted)

    print(f"{(time() - start) * 1000 / 8:.2f} ms")


pipeline()