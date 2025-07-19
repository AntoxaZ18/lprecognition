import json
import os
import sys
from queue import Queue
import psutil

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QListWidget,
    QListWidgetItem
)

from inference_engine import Inference, ModelLoadFS
from pipeline import VideoPipeLine
from render import Render
from saver import SaverEngine, SQLSaver, SQLSession, SQLResults


class cQLineEdit(QLineEdit):
    """
    clickable qLineEdit
    """

    clicked = pyqtSignal()

    def __init__(self, widget):
        super().__init__(widget)

    def mousePressEvent(self, QMouseEvent):
        self.clicked.emit()


class PipeShow:
    def __init__(
        self,
        video_src: str,
        inference: Inference,
        config: dict,
        result_queue: Queue,
        callback,
    ):
        self.render_frame_queue = Queue()
        self.video_pipe = VideoPipeLine(
            video_src,
            inference,
            config,
            output_queue=self.render_frame_queue,
            result_queue=result_queue,
        )
        self.renderer = Render(self.render_frame_queue, 1920 // 2, 1080 // 2)

        self.renderer.update_pixmap_signal.connect(callback)

    def start(self):
        self.video_pipe.start()
        self.renderer.start()

    def stop(self):
        self.video_pipe.stop()
        self.renderer.stop()


class VideoWidget(QWidget):
    FIXED_WIDTH = 640
    FIXED_HEIGHT = 360

    def __init__(self, parent=None):
        super().__init__(parent)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(self.FIXED_WIDTH, self.FIXED_HEIGHT)
        self.image_label.setStyleSheet("background-color: black;")  # Опционально

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.qt_img = None

    def update_frame(self, cv_img):
        self.qt_img = QPixmap.fromImage(cv_img)
        self.image_label.setPixmap(
            self.qt_img.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        if self.qt_img:
            self.image_label.setPixmap(
                self.qt_img.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        super().resizeEvent(event)


class VideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("License Plate recognition")
        self.setGeometry(100, 100, 1920 // 2, 1080 // 2)
        self.width = 800
        self.height = 600

        self.timer = QTimer()
        self.result_timer = QTimer()

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.video_grid = QGridLayout()
        self.video_widgets = []  # список VideoWidget
        self.video_threads = []  # список потоков

        self.file_label = QLabel(self.central_widget)
        self.file_label.setText("Видеофайл")
        self.video_source = cQLineEdit(self.central_widget)
        self.video_source.clicked.connect(self.choose_source)

        self.start_button = QPushButton("Старт")
        self.start_button.clicked.connect(self.start_video)
        self.stop_button = QPushButton("Стоп")
        self.stop_button.clicked.connect(self.stop_video)

        self.perf_label = QLabel(self)
        self.perf_label.setText("0 FPS")

        self.video_fps = QLabel(self)
        self.video_fps.setText("0 FPS")

        self.cpu_load = QLabel(self)
        self.cpu_load.setText(f"load: {psutil.cpu_percent(interval=None):.2f} %")

        self.event_list = QListWidget(self.central_widget)

        control_layout = QVBoxLayout()
        control_layout.addWidget(self.file_label)
        control_layout.addWidget(self.video_source)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.perf_label)
        control_layout.addWidget(self.video_fps)
        control_layout.addWidget(self.cpu_load)

        control_layout.addWidget(QLabel("События:"))
        control_layout.addWidget(self.event_list)
        control_layout.addStretch()

        # ------------------------------------------------
        control_widget = QWidget()
        control_widget.setLayout(control_layout)
        control_widget.setMaximumWidth(200)

        # --- Основной layout ---
        main_layout = QHBoxLayout()
        main_layout.addLayout(self.video_grid)
        main_layout.addWidget(control_widget)

        self.central_widget.setLayout(main_layout)

        self.result_queue = Queue()
        self.inf = Inference(ModelLoadFS("./models"))
        self.sql_session = SQLSession("sqlite:///events.db")
        self.saver = SQLSaver(self.sql_session())
        self.show_results = SQLResults(self.sql_session(), last=20)
        self.saver_handler = SaverEngine(self.saver, self.result_queue)

        self.timer.timeout.connect(self.update_fps)
        self.timer.start(1000)  # время обновления FPS in ms

        self.result_timer.timeout.connect(self.refresh_results)
        self.result_timer.start(1000)

    def refresh_results(self):

        self.event_list.clear()
        for plate, timestamp in self.show_results():
            item = QListWidgetItem(f"{plate} {timestamp.strftime('%H:%M:%S')}")
            self.event_list.addItem(item)



    def start_video(self):
        if not self.video_source.text():
            QMessageBox.warning(self, "Warning", "Нужно выбрать файл")
            return

        # Создаем виджет для видео
        video_widget = VideoWidget()
        self.video_widgets.append(video_widget)
        row = len(self.video_widgets) - 1
        self.video_grid.addWidget(video_widget, row // 2, row % 2)  # 2 колонки

        config = json.load(open("pipe_cfg.json", "r", encoding="utf-8"))
        pipe = PipeShow(
            self.video_source.text(),
            self.inf,
            config,
            self.result_queue,
            video_widget.update_frame,
        )

        self.video_threads.append(pipe)

        pipe.start()

    def stop_video(self):
        if self.video_threads:
            for thread in self.video_threads:
                thread.stop()
        self.video_threads.clear()

    def choose_model_folder(self):
        options = QFileDialog.Option.DontUseNativeDialog
        folder_path = QFileDialog.getExistingDirectory(
            self, "Выберите папку", options=options
        )
        if folder_path:
            self.model_file.clear()
            onnx_files = [i for i in os.listdir(folder_path) if i.endswith(".onnx")]
            self.model_file.addItems(onnx_files)
            self.model_folder.setText(folder_path)

    def choose_source(self):
        options = QFileDialog.Option.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл",
            "",
            "Все файлы (*);;Видео файлы (*.mp4)",
            options=options,
        )

        if file_name:
            self.video_source.setText(file_name)

    def update_fps(self):

        self.cpu_load.setText(f"load: {psutil.cpu_percent(interval=None):.2f} %")


        if self.video_threads:
            self.perf_label.setText(f"inference: {self.inf.batches_per_second} BPS")
            total_fps = sum([thread.renderer.fps for thread in self.video_threads])
            self.video_fps.setText(f"Rendering: {total_fps:.2f} FPS")

    def refresh_image(self, cv_img):
        self.qt_img = QPixmap.fromImage(cv_img)
        self.image_label.setPixmap(self.qt_img)
        original_size = self.qt_img.size()

        # self.scale_image()
        self.setMinimumSize(original_size.width(), original_size.height())

    def closeEvent(self, event):
        if self.video_threads:
            for thread in self.video_threads:
                thread.stop()
        self.video_threads.clear()

        if self.saver_handler:
            self.saver_handler.stop()

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec())
