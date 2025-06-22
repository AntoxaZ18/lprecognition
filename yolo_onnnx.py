from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from time import time
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

# def get_providers():
#     providers = [i for i in ort.get_available_providers() if any(val in i for val in ('CUDA', 'CPU'))]

#     modes = {
#         'CUDAExecutionProvider': 'gpu',
#         'CPUExecutionProvider': 'cpu'
#     }
#     providers = [modes.get(i) for i in providers]
#     return providers


class YoloONNX:
    def __init__(
        self, path: str, session_options=None, device="cpu", threads=1, confidence=0.2, classes=None
    ) -> None:
        if classes is None:
            raise ValueError("Provide classes as list of names")

        sess_options = ort.SessionOptions()

        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.inter_op_num_threads = 3

        sess_providers = ['CPUExecutionProvider']
        if device == 'gpu' or device == 0:
            sess_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.mode = 'gpu'
        else:
            self.mode = 'cpu'

        self.session = ort.InferenceSession(
            path, providers=sess_providers, sess_options=sess_options
        )  #'CUDAExecutionProvider',
        model_inputs = self.session.get_inputs()
        input_shape = model_inputs[0].shape

        self.input_width = 320
        self.input_height = 320

        self.iou = 0.4
        self.confidence_thres = confidence
        self.input_size = (320, 320)
        self.classes = classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.executor = ThreadPoolExecutor(max_workers=threads)

    def _image_preprocess(self, bgr_frame) -> np.ndarray:
        """image preprocessing
        bgr_frame - image in bgr format
        including resizing to yolo input shape
        add batch dimension and normalized to 0...1 range
        convert from image to tensor view
        #"""
        self.img_height, self.img_width = bgr_frame.shape[:2]
        rgb_img = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        img, pad = self.letterbox(rgb_img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data, pad

    def letterbox(
        self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Resize and reshape images while maintaining aspect ratio by adding padding.

        Args:
            img (np.ndarray): Input image to be resized.
            new_shape (Tuple[int, int]): Target shape (height, width) for the image.

        Returns:
            (np.ndarray): Resized and padded image.
            (Tuple[int, int]): Padding values (top, left) applied to the image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (
            (new_shape[1] - new_unpad[0]) / 2,
            (new_shape[0] - new_unpad[1]) / 2,
        )  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return img, (top, left)

    def draw_detections(
        self, img: np.ndarray, box: List[float], score: float, class_id: int
    ) -> None:
        """
        Draw bounding boxes and labels on the input image based on the detected objects.

        Args:
            img (np.ndarray): The input image to draw detections on.
            box (List[float]): Detected bounding box coordinates [x, y, width, height].
            score (float): Confidence score of the detection.
            class_id (int): Class ID for the detected object.
        """
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img,
            (label_x, label_y - label_height),
            (label_x + label_width, label_y + label_height),
            color,
            cv2.FILLED,
        )

        # Draw the label text on the image
        cv2.putText(
            img,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    def postprocess(
        self, input_image: np.ndarray, output: List[np.ndarray], pad: Tuple[int, int]
    ) -> np.ndarray:
        """
        Perform post-processing on the model's output to extract and visualize detections.

        This method processes the raw model output to extract bounding boxes, scores, and class IDs.
        It applies non-maximum suppression to filter overlapping detections and draws the results on the input image.

        Args:
            input_image (np.ndarray): The input image.
            output (List[np.ndarray]): The output arrays from the model.
            pad (Tuple[int, int]): Padding values (top, left) used during letterboxing.

        Returns:
            (np.ndarray): The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape

        outputs = np.transpose(np.squeeze(output[0]))

        # Calculate the scaling factors for the bounding box coordinates
        input_height = self.input_size[0]
        input_width = self.input_size[0]

        gain = np.float32(
            min(input_height / self.img_height, input_width / self.img_width)
        )
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]

        # find max score for prediction
        max_scores = np.amax(outputs[:, 4:], axis=1)
        # filter prediction with result more than confidence thresh
        results = outputs[max_scores > self.confidence_thres]
        # find class ids
        class_ids = np.argmax(results[:, 4:], axis=1).tolist()
        scores = np.amax(results[:, 4:], axis=1).tolist()

        # convert coordinates of boxes
        x = results[:, 0]
        y = results[:, 1]
        w = results[:, 2]
        h = results[:, 3]

        left = ((x - w / 2) / gain).astype(int)
        top = ((y - h / 2) / gain).astype(int)
        width = (w / gain).astype(int)
        height = (h / gain).astype(int)

        boxes = np.column_stack((left, top, width, height)).tolist()

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou)

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)

        # Return the modified input image
        return input_image

    def __call__(self, images: np.ndarray) -> np.ndarray:

        if self.mode == 'gpu':
            return self.call_gpu(images)
        else:
            return self.call_cpu(images)

    def call_gpu(self, images: List[np.ndarray]):

        tensor_images = [self._image_preprocess(image) for image in images]

        output_name = self.session.get_outputs()[0].name
        input_name = self.session.get_inputs()[0].name

        imgs = [img for (img, _) in tensor_images]
        pads = [pad for (_, pad) in tensor_images]

        batch = np.concatenate(imgs, axis=0)

        outputs = self.session.run([output_name], {input_name: batch})

        predictions = outputs[0]

        results = [
            self.postprocess(image, np.expand_dims(predictions[idx], axis=0), pad)
            for image, idx, pad in zip(images, iter(range(predictions.shape[0])), pads)
        ]

        return results

    def cpu_run_session(self, image, output_name, input_name):
        tensor_image, pad = self._image_preprocess(image)
        predictions = self.session.run([output_name], {input_name: tensor_image})

        return predictions, pad

    def call_cpu(self, images: List[np.ndarray]):
        output_name = self.session.get_outputs()[0].name
        input_name = self.session.get_inputs()[0].name

        futures = [
            self.executor.submit(self.cpu_run_session, image, output_name, input_name)
            for image in images
        ]
        wait(futures, return_when=ALL_COMPLETED)

        results = [f.result() for f in futures]

        model_outputs = [outputs for (outputs, _) in results]
        pads = [pad for (_, pad) in results]

        results = [
            self.postprocess(image, output, pad)
            for output, image, pad in zip(model_outputs, images, pads)
        ]

        return results