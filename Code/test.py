import torch
import torchvision
import torchaudio
import cv2



# 打印版本号
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("Torchaudio version:", torchaudio.__version__)
print("OpenCV version:", cv2.__version__)

# 检查设备支持
print("CUDA available:", torch.cuda.is_available())
if torch.backends.mps.is_available():
    print("MPS (Metal Performance Shaders) available:", torch.backends.mps.is_available())

from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

class DragDropTest(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setAcceptDrops(True)
        self.setGeometry(100, 100, 400, 300)
        self.setWindowTitle("拖放测试")

        self.label = QLabel("拖动文件到此区域", self)
        self.label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.label.setText("释放文件上传")
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.label.setText("拖动文件到此区域")

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_paths = [url.toLocalFile() for url in event.mimeData().urls()]
            self.label.setText(f"文件路径: {file_paths[0]}")
        else:
            self.label.setText("未识别的拖放事件")

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    demo = DragDropTest()
    demo.show()
    sys.exit(app.exec_())