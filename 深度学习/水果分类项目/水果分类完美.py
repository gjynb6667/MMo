import os
import random
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QMessageBox, QScrollArea, QFrame, QGridLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from  ResNet网络结构 import  ResNet,BasicBlock
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 类别映射
CLASS_NAMES = ['abiu', 'acai', 'acerola', 'ackee']
CLASS_MAPPING = {0: 'abiu', 1: 'acai', 2: 'acerola', 3: 'ackee'}


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


set_seed(0)

# 设置设备
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CPU is available. Using CPU.")

# 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# 加载模型
def load_model(model_path, num_classes=4):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"成功加载模型: {model_path}")
    else:
        print(f"警告: 模型文件不存在 {model_path}")
    return model


# 加载两个模型
model3 = load_model("./model/shuiguo3.pth")
model4 = load_model("./model/shuiguo4.pth")


class MatplotlibCanvas(FigureCanvas):
    """用于在PyQt中显示matplotlib图表的画布"""

    def __init__(self, parent=None, width=5, height=4):
        self.fig = Figure(figsize=(width, height))
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.axes = self.fig.add_subplot(111)

    def plot_confusion_matrix(self, conf_matrix, title="Confusion Matrix"):
        """绘制混淆矩阵"""
        self.axes.clear()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    cbar=False, ax=self.axes, xticklabels=CLASS_NAMES,
                    yticklabels=CLASS_NAMES)
        self.axes.set_xlabel("Predicted")
        self.axes.set_ylabel("True")
        self.axes.set_title(title)
        self.fig.tight_layout()
        self.draw()


def get_label_from_filename(filename):
    """从文件名中提取真实标签"""
    filename_lower = filename.lower()

    if 'abiu' in filename_lower:
        return 0  # abiu
    elif 'acai' in filename_lower:
        return 1  # acai
    elif 'acerola' in filename_lower:
        return 2  # acerola
    elif 'ackee' in filename_lower:
        return 3  # ackee
    else:
        return -1  # 无法识别
class FruitClassifierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("水果分类系统")
        self.setGeometry(300, 300, 1400, 800)

        # 存储当前选择的图片路径
        self.selected_images = []

        # 创建中央窗口
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # 标题
        title_label = QLabel("水果分类系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        main_layout.addWidget(title_label)

        # 按钮布局
        button_layout = QHBoxLayout()
        self.single_btn = QPushButton("单图片预测")
        self.multi_btn = QPushButton("多图片预测")
        self.evaluate_btn = QPushButton("模型评估（测试集）")
        self.single_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        self.multi_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        self.evaluate_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        self.single_btn.clicked.connect(self.single_image_prediction)
        self.multi_btn.clicked.connect(self.multi_image_prediction)
        self.evaluate_btn.clicked.connect(self.evaluate_on_testset)
        button_layout.addWidget(self.single_btn)
        button_layout.addWidget(self.multi_btn)
        button_layout.addWidget(self.evaluate_btn)
        main_layout.addLayout(button_layout)

        # 创建滚动区域用于显示结果
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.result_layout = QGridLayout()
        scroll_widget.setLayout(self.result_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        # 初始化结果显示区域
        self.result_widgets = []

    def preprocess_image(self, image_path):
        """预处理单张图片"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = data_transforms(image)
            image = image.unsqueeze(0)  # 添加batch维度
            return image.to(device)
        except Exception as e:
            print(f"图片预处理错误: {e}")
            return None

    def predict_images(self, image_paths):
        """对多张图片进行预测"""
        results_model3 = []
        results_model4 = []

        for path in image_paths:
            image_tensor = self.preprocess_image(path)
            if image_tensor is None:
                continue

            # 模型3预测
            with torch.no_grad():
                output3 = model3(image_tensor)
                _, predicted3 = torch.max(output3, 1)
                confidence3 = torch.nn.functional.softmax(output3, dim=1)
                confidence3_value = confidence3[0][predicted3].item()

            # 模型4预测
            with torch.no_grad():
                output4 = model4(image_tensor)
                _, predicted4 = torch.max(output4, 1)
                confidence4 = torch.nn.functional.softmax(output4, dim=1)
                confidence4_value = confidence4[0][predicted4].item()

            results_model3.append({
                'path': path,
                'pred_class': predicted3.item(),
                'pred_name': CLASS_MAPPING[predicted3.item()],
                'confidence': confidence3_value
            })
            results_model4.append({
                'path': path,
                'pred_class': predicted4.item(),
                'pred_name': CLASS_MAPPING[predicted4.item()],
                'confidence': confidence4_value
            })

        return results_model3, results_model4

    def calculate_accuracy_with_labels(self, results_model3, results_model4, true_labels):
        """使用真实标签计算准确率"""
        correct3 = 0
        correct4 = 0
        valid_count = 0

        for i, result in enumerate(results_model3):
            if true_labels[i] != -1:  # 只计算能识别标签的图片
                valid_count += 1
                if result['pred_class'] == true_labels[i]:
                    correct3 += 1

                if results_model4[i]['pred_class'] == true_labels[i]:
                    correct4 += 1

        # 计算准确率百分比
        accuracy3 = (correct3 / valid_count * 100) if valid_count > 0 else 0
        accuracy4 = (correct4 / valid_count * 100) if valid_count > 0 else 0

        # 统计各类别预测数量
        pred_counts3 = {i: 0 for i in range(4)}
        pred_counts4 = {i: 0 for i in range(4)}

        for result in results_model3:
            pred_counts3[result['pred_class']] += 1
        for result in results_model4:
            pred_counts4[result['pred_class']] += 1

        # 打印详细信息
        print(f"\n预测结果统计:")
        print(f"有效图片数: {valid_count}")
        print(f"模型3正确数: {correct3}/{valid_count} (准确率: {accuracy3:.2f}%)")
        print(f"模型4正确数: {correct4}/{valid_count} (准确率: {accuracy4:.2f}%)")
        print(f"模型3预测分布: {pred_counts3}")
        print(f"模型4预测分布: {pred_counts4}")

        return accuracy3, accuracy4, pred_counts3, pred_counts4

    def evaluate_on_testset(self):
        """在测试集上评估模型"""
        # 将您的评估代码移到这里
        print("开始在测试集上评估模型...")

        # 重新加载数据集
        data_transforms_dict  = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
            "val": transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        }

        train_dir = './dataset/shuiguo/train'
        val_dir = './dataset/shuiguo/val'

        train_dataset = datasets.ImageFolder(
            train_dir,
            transform=data_transforms_dict["train"]
        )

        val_dataset = datasets.ImageFolder(
            val_dir,
            transform=data_transforms_dict["val"]
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

        # 获取类别名称
        class_names = train_dataset.classes

        # 评估模型3
        model3.eval()
        true_labels = []
        pred_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model3(images)
                _, predicted = torch.max(outputs, 1)
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(predicted.cpu().numpy())

        # 计算指标
        accuracy = 100 * sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / len(true_labels)
        precision_per_class = precision_score(true_labels, pred_labels, average=None)
        recall_per_class = recall_score(true_labels, pred_labels, average=None)
        f1_per_class = f1_score(true_labels, pred_labels, average=None)

        print(f"\n各类别指标:")
        print(f"{'类别':<15} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
        print("-" * 45)
        for i, class_name in enumerate(class_names):
            print(
                f"{class_name:<15} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} {f1_per_class[i]:<10.4f}")

        conf = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Model 3 - Confusion Matrix (Accuracy: {accuracy:.2f}%)")
        plt.show()

        QMessageBox.information(self, "评估完成", f"模型3测试集准确率: {accuracy:.2f}%")

    def display_results_with_accuracy(self, image_paths, results_model3, results_model4,
                                      accuracy3, accuracy4, true_labels):
        """显示带准确率的结果"""
        # 清除之前的结果
        for widget in self.result_widgets:
            widget.deleteLater()
        self.result_widgets.clear()

        # 清除布局
        for i in reversed(range(self.result_layout.count())):
            self.result_layout.itemAt(i).widget().setParent(None)

        # 显示每张图片的预测结果
        for idx, (path, result3, result4) in enumerate(zip(image_paths, results_model3, results_model4)):
            # 创建结果框架
            frame = QFrame()
            frame.setFrameStyle(QFrame.Box)
            frame.setStyleSheet("QFrame { border: 2px solid #ccc; border-radius: 5px; margin: 5px; }")
            layout = QVBoxLayout()

            # 图片显示
            pixmap = QPixmap(path)
            pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_label = QLabel()
            img_label.setPixmap(pixmap)
            img_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(img_label)

            # 图片名
            path_label = QLabel(os.path.basename(path))
            path_label.setAlignment(Qt.AlignCenter)
            path_label.setStyleSheet("font-size: 10px;")
            layout.addWidget(path_label)

            if true_labels[idx] != -1:
                true_label_text = f"真实: {CLASS_MAPPING[true_labels[idx]]}"
                true_label = QLabel(true_label_text)
                true_label.setAlignment(Qt.AlignCenter)
                true_label.setStyleSheet("color: green; font-weight: bold;")
                layout.addWidget(true_label)

            # 模型3预测结果
            result3_text = f"模型3: {result3['pred_name']} ({result3['confidence']:.2%})"
            result3_label = QLabel(result3_text)
            result3_label.setAlignment(Qt.AlignCenter)
            if true_labels[idx] != -1:
                if result3['pred_class'] == true_labels[idx]:
                    result3_label.setStyleSheet("color: blue; font-weight: bold; background-color: #d4edda;")
                else:
                    result3_label.setStyleSheet("color: red; font-weight: bold; background-color: #f8d7da;")
            else:
                result3_label.setStyleSheet("color: blue; font-weight: bold;")
            layout.addWidget(result3_label)

            # 模型4预测结果
            result4_text = f"模型4: {result4['pred_name']} ({result4['confidence']:.2%})"
            result4_label = QLabel(result4_text)
            result4_label.setAlignment(Qt.AlignCenter)
            if true_labels[idx] != -1:
                if result4['pred_class'] == true_labels[idx]:
                    result4_label.setStyleSheet("color: green; font-weight: bold; background-color: #d4edda;")
                else:
                    result4_label.setStyleSheet("color: orange; font-weight: bold; background-color: #fff3cd;")
            else:
                result4_label.setStyleSheet("color: green; font-weight: bold;")
            layout.addWidget(result4_label)

            frame.setLayout(layout)
            self.result_layout.addWidget(frame, idx // 3, idx % 3)
            self.result_widgets.append(frame)

        # 添加准确率显示
        row_offset = (len(image_paths) + 2) // 3

        # 模型3准确率
        acc_frame3 = QFrame()
        acc_frame3.setStyleSheet("QFrame { background-color: #e3f2fd; border-radius: 5px; margin: 5px; }")
        acc_layout3 = QVBoxLayout()
        acc_title3 = QLabel("模型3 (shuiguo3.pth)")
        acc_title3.setAlignment(Qt.AlignCenter)
        acc_title3.setStyleSheet("font-weight: bold; font-size: 12px;")
        acc_value3 = QLabel(f"准确率: {accuracy3:.2f}%")
        acc_value3.setAlignment(Qt.AlignCenter)
        acc_value3.setStyleSheet("font-size: 18px; font-weight: bold; color: blue;")
        acc_layout3.addWidget(acc_title3)
        acc_layout3.addWidget(acc_value3)
        acc_frame3.setLayout(acc_layout3)
        self.result_layout.addWidget(acc_frame3, row_offset, 0)
        self.result_widgets.append(acc_frame3)

        # 模型4准确率
        acc_frame4 = QFrame()
        acc_frame4.setStyleSheet("QFrame { background-color: #e8f5e8; border-radius: 5px; margin: 5px; }")
        acc_layout4 = QVBoxLayout()
        acc_title4 = QLabel("模型4 (shuiguo4.pth)")
        acc_title4.setAlignment(Qt.AlignCenter)
        acc_title4.setStyleSheet("font-weight: bold; font-size: 12px;")
        acc_value4 = QLabel(f"准确率: {accuracy4:.2f}%")
        acc_value4.setAlignment(Qt.AlignCenter)
        acc_value4.setStyleSheet("font-size: 18px; font-weight: bold; color: green;")
        acc_layout4.addWidget(acc_title4)
        acc_layout4.addWidget(acc_value4)
        acc_frame4.setLayout(acc_layout4)
        self.result_layout.addWidget(acc_frame4, row_offset, 1)
        self.result_widgets.append(acc_frame4)

    def single_image_prediction(self):
        """单图片预测"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            self.selected_images = [file_path]

            # 从文件名获取真实标签
            filename = os.path.basename(file_path)
            true_label = get_label_from_filename(filename)

            # 进行预测
            results_model3, results_model4 = self.predict_images(self.selected_images)

            # 计算准确率
            accuracy3, accuracy4, pred_counts3, pred_counts4 = self.calculate_accuracy_with_labels(
                results_model3, results_model4, [true_label]
            )

            # 显示结果（带准确率）
            self.display_results_with_accuracy(
                self.selected_images, results_model3, results_model4,
                accuracy3, accuracy4, [true_label]
            )

            QMessageBox.information(self, "预测完成", f"已完成对单张图片的预测")
        else:
            QMessageBox.warning(self, "警告", "未选择图片")

    def multi_image_prediction(self):
        """多图片预测（2张以上）"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择多张图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if len(file_paths) >= 2:
            self.selected_images = file_paths

            # 从文件名获取真实标签
            true_labels = []
            for path in file_paths:
                filename = os.path.basename(path)  # 获取文件名，如 "abiu1.jpg"
                label = get_label_from_filename(filename)
                true_labels.append(label)

            # 进行预测
            results_model3, results_model4 = self.predict_images(self.selected_images)

            # 计算准确率（传入真实标签）
            accuracy3, accuracy4, pred_counts3, pred_counts4 = self.calculate_accuracy_with_labels(
                results_model3, results_model4, true_labels
            )

            # 显示结果（带准确率）
            self.display_results_with_accuracy(
                self.selected_images, results_model3, results_model4,
                accuracy3, accuracy4, true_labels
            )

            QMessageBox.information(self, "预测完成",
                                    f"已完成对 {len(file_paths)} 张图片的预测\n模型3准确率: {accuracy3:.2f}%\n模型4准确率: {accuracy4:.2f}%")
        elif len(file_paths) > 0:
            QMessageBox.warning(self, "警告", "请至少选择2张图片进行多图片预测")
        else:
            QMessageBox.warning(self, "警告", "未选择图片")



def main():
    app = QApplication(sys.argv)
    window = FruitClassifierGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":


    main()
