import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QLabel, QPushButton, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import soundfile as sf
import librosa
import librosa.display

class GenderRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.set_style()

    def init_ui(self):
        """初始化用户界面"""
        #设置字体
        from matplotlib import rcParams
        rcParams['font.sans-serif'] = ['SimHei']
        rcParams['axes.unicode_minus'] = False
        self.setWindowTitle("基于基频分析的性别识别系统")
        self.setGeometry(100, 100, 1000, 800)

        # 主部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 标题
        self.title_label = QLabel("请选择音频文件进行性别识别")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font: bold 18px; margin: 20px 0;")
        layout.addWidget(self.title_label)

        # 文件选择按钮
        self.file_btn = QPushButton("选择音频文件")
        self.file_btn.clicked.connect(self.load_audio)
        self.file_btn.setStyleSheet(
            "QPushButton {"
            "background-color: #4CAF50;"
            "color: white;"
            "padding: 12px 24px;"
            "border: none;"
            "border-radius: 4px;"
            "font: bold 14px;"
            "}"
            "QPushButton:hover {background-color: #45a049;}"
        )
        layout.addWidget(self.file_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # 结果显示
        self.result_label = QLabel("等待分析...")
        self.result_label.setStyleSheet("font: 14px; color: #666; margin: 20px 0;")
        layout.addWidget(self.result_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # 频谱图显示区域
        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def set_style(self):
        """设置全局样式"""
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                font-family: Arial;
            }
            QLabel {
                color: #333;
            }
        """)

    def load_audio(self):
        """加载音频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频文件",
            "",
            "音频文件 (*.wav *.mp3)"
        )
        if not file_path:
            return

        try:
            y, sr = librosa.load(file_path, sr=None)
            self.analyze_audio(y, sr)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"文件加载失败：{str(e)}")

    def analyze_audio(self, y, sr):
        """分析音频并显示结果"""
        # 清除之前的绘图
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        try:
            # 基频分析
            f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=800, sr=sr)
            f0 = f0[voiced_flag]
            
            if len(f0) == 0:
                self.show_result("无法检测到有效基频", "#ff0000")
                return

            avg_f0 = np.nanmean(f0)
            gender = "女性" if avg_f0 > 165 else "男性"
            self.show_result(f"基频平均值：{avg_f0:.1f}Hz → {gender}", 
                            "#2196F3" if gender == "男性" else "#E91E63")

            # 绘制频谱图
            D = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            img = librosa.display.specshow(
                S_db,
                sr=sr,
                x_axis='time',
                y_axis='log',
                ax=ax
            )
            self.figure.colorbar(img, ax=ax, format="%+2.0f dB")
            ax.set(title='频谱图（对数频率轴）')
            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "分析错误", str(e))

    def show_result(self, text, color):
        """更新结果标签"""
        self.result_label.setText(text)
        self.result_label.setStyleSheet(f"""
            font: bold 16px; 
            color: {color};
            margin: 20px 0;
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GenderRecognitionApp()
    window.show()
    sys.exit(app.exec())