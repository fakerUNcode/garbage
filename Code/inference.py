import mysql.connector
import torch
import cv2
from PyQt5.QtCore import QTimer

from config import DefaultConfig
from models.mobilenetv3 import MobileNetV3_Small
from torchvision import transforms
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QLineEdit, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QMovie


class RecycleApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initDatabase()

    def initUI(self):
        self.setWindowTitle("垃圾分类与奖励系统")
        self.setGeometry(100, 100, 800, 600)

        self.userLabel = QLabel("用户:")
        self.userInput = QLineEdit()

        self.passwordLabel = QLabel("密码:")
        self.passwordInput = QLineEdit()
        self.passwordInput.setEchoMode(QLineEdit.Password)

        self.loginButton = QPushButton("登录")
        self.loginButton.clicked.connect(self.handleLogin)

        self.imageLabel = QLabel("选择图片分类:")
        self.uploadButton = QPushButton("上传图片")
        self.uploadButton.clicked.connect(self.uploadImage)

        self.resultLabel = QLabel("分类结果将在这里显示")

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.userLabel)
        self.layout.addWidget(self.userInput)
        self.layout.addWidget(self.passwordLabel)
        self.layout.addWidget(self.passwordInput)
        self.layout.addWidget(self.loginButton)
        self.layout.addWidget(self.imageLabel)
        self.layout.addWidget(self.uploadButton)
        self.layout.addWidget(self.resultLabel)

        centralWidget = QLabel()
        centralWidget.setLayout(self.layout)
        self.setCentralWidget(centralWidget)

    def initDatabase(self):
        """初始化数据库连接"""
        self.conn = mysql.connector.connect(
            host="localhost",
            user="root",              # 替换为您的 MySQL 用户名
            password="Faker666!", # 替换为您的 MySQL 密码
            database="recycle"
        )
        self.cursor = self.conn.cursor()

    def authenticateUser(self, username, password):
        """验证用户登录"""
        self.cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = self.cursor.fetchone()
        return user is not None

    def updateRecycleCount(self, username):
        """更新用户回收次数，并检查是否需要触发奖励动画"""
        self.cursor.execute("SELECT recycle_count FROM users WHERE username = %s", (username,))
        count = self.cursor.fetchone()[0] + 1
        self.cursor.execute("UPDATE users SET recycle_count = %s WHERE username = %s", (count, username))
        self.conn.commit()

        if count % 5 == 0:
            print(f"条件触发：回收次数 = {count}")
            self.showRewardAnimation()
        else:
            print(f"条件未触发：回收次数 = {count}")

    def showRewardAnimation(self):
        """显示奖励动画"""
        # 使用实例变量存储窗口，防止被垃圾回收
        self.rewardWindow = QLabel()
        self.rewardWindow.setWindowTitle("恭喜你回收了五次垃圾，成功助力环保！")
        self.rewardWindow.setGeometry(620, 620, 620, 620)

        movie = QMovie("reward_animation.gif")
        if not movie.isValid():
            print("奖励动画文件无效或路径错误")
            return

        print("奖励动画文件有效，启动播放")
        self.rewardWindow.setMovie(movie)
        movie.start()

        # 设置动画窗口关闭时间
        QTimer.singleShot(5000, self.rewardWindow.close)  # 5 秒后自动关闭窗口
        self.rewardWindow.show()
        print("奖励动画窗口已显示")

    def handleLogin(self):
        """处理登录"""
        username = self.userInput.text()
        password = self.passwordInput.text()
        if self.authenticateUser(username, password):
            QMessageBox.information(self, "成功", "登录成功！")
        else:
            QMessageBox.warning(self, "失败", "用户名或密码错误！")

    def uploadImage(self):
        """上传图片并分类"""
        fileName, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if fileName:
            image = cv2.imread(fileName)
            result, group = self.inferImage(image)
            self.resultLabel.setText(f"分类结果: {result}, 分组: {group}")
            self.updateRecycleCount(self.userInput.text())

    def inferImage(self, cvImg):
        """推理图片分类"""
        image = cvImgToTensor(cvImg)
        image = image.to(device=device)
        result = MyModel(image)
        _, predicted = torch.max(result, 1)
        predicted = predicted.item()
        return index_to_class[predicted], index_to_group[predicted]


# 模型加载与配置
def cvImgToTensor(cvImg):
    image = cvImg.copy()
    height, width, channel = image.shape
    ratio = 224 / min(height, width)
    image = cv2.resize(image, None, fx=ratio, fy=ratio)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    if image is not None:
        image = image[:, :, (2, 1, 0)]
        image = transform(image)
        image.unsqueeze_(0)

    return image


# 加载模型
DataSetInfo = torch.load(DefaultConfig.DataSetInfoPath)  # 使用您的模型配置路径
index_to_class = DataSetInfo['index_to_class']
index_to_group = DataSetInfo['index_to_group']
MyModel = MobileNetV3_Small(DataSetInfo["class_num"])
device = torch.device('cuda') if DefaultConfig.InferWithGPU else torch.device('cpu')
MyModel.load_state_dict(torch.load(DefaultConfig.CkptPath, map_location=torch.device('cpu'))['state_dict'])
if DefaultConfig.InferWithGPU:
    MyModel.cuda()
else:
    MyModel.cpu()
MyModel.eval()

if __name__ == "__main__":
    app = QApplication([])
    mainWindow = RecycleApp()
    mainWindow.show()
    app.exec_()
