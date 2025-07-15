import sys
from inference_engine import Inference, ModelLoadFS

from time import sleep, time


from pipeline import VideoPipeLine



inf = Inference(ModelLoadFS('./models'))



threads_num = 1

# threads = [VideoPipeLine("video.mp4", inf, yolo_shape=Model_shape) for _ in range(threads_num)]

# for t in threads:
#     t.start()
#     sleep(0.01)

config = {
    "1": {
        "name": "yolo_lp",
        # "model": "ex6_c3k2_light_640x640.onnx",
        "model": "ex8_c3k2_light_320_nwd_320x320.onnx",

        "args": {"bgr": True}
    },
    "2": {
        "name": "lpr_recognition",
        "model": "stn_lpr_opt_final_94x24.onnx",
    }
}


video_stream = VideoPipeLine("video.mp4", inf, config)
video_stream.start()
# video_stream_2.start()

while True:
    sleep(1)
    print(inf.batches_per_second())