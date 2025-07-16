import PyInstaller.__main__
from pathlib import Path
import shutil
import os

HERE = Path(__file__).parent.absolute()
path_to_main = str(HERE / "app.py")


# &py $py_ver -m PyInstaller app.py `
#     --add-data $sip_pyd `
#     --add-data $qt_folder `
#     --add-binary="icon.ico;." `
#     --clean --name=onnx_rt_viewer -w -F --ico icon.ico


def install():
    icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")

    if not os.path.isfile(icon_path):
        raise FileNotFoundError(f"Файл иконки {icon_path} не найден")

    PyInstaller.__main__.run(
        [
            path_to_main,
            "--clean",
            "-n onnx_rt_viewer",
            f"--add-binary={icon_path};.",
            f"-i{icon_path}",
            "--hidden-import=onnxruntime.capi._pybind_state", 
            "--hidden-import=onnxruntime.capi.onnxruntime_pybind11_state",
            "--onefile",
            "--windowed",            
        ]
    )
    shutil.rmtree("./build")


if __name__ == "__main__":
    install()