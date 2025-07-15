import sys
from inference_engine import Inference, ModelLoadFS

from time import sleep, time


from pipeline import VideoPipeLine


Model_shape = (320, 320)

inf = Inference(ModelLoadFS('./models'))
inf.create_session('ex6_c3k2_light_640x640.onnx', "yolo_lp", args = {"model_shape": Model_shape, "bgr":True})
inf.create_session('stn_lpr_opt_final_94x24.onnx', "lpr_recognition", args = {"model_shape":(94, 24)})


threads_num = 1

# threads = [VideoPipeLine("video.mp4", inf, yolo_shape=Model_shape) for _ in range(threads_num)]

# for t in threads:
#     t.start()
#     sleep(0.01)

video_stream = VideoPipeLine("video.mp4", inf, yolo_shape=Model_shape)
# video_stream_2 = VideoPipeLine("video.mp4", yolo_shape=Model_shape)


# batch = {f"{x}": f"{x}" for x in range(24)}

# print(batch)

# resampled = video_stream.resample_images(batch, 8)

# print(resampled)

video_stream.start()
# video_stream_2.start()

while True:
    sleep(1)
    print(inf.batches_per_second())