import sys
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import pyaudio
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFileDialog, QMessageBox, QSizePolicy)
from PyQt6.QtCore import Qt, QTimer
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class GenderRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # 音频参数
        self.CHUNK = 1024  # 每次读取的音频块大小
        self.FORMAT = pyaudio.paInt16  # 音频格式
        self.CHANNELS = 1  # 单声道
        self.RATE = 16000  # 采样率
        self.recording = False
        self.audio_data = np.array([], dtype=np.float32)
        self.p = pyaudio.PyAudio()
        
        self.init_ui()
        self.set_style()

    def init_ui(self):
        """初始化用户界面"""
        #设置字体
        from matplotlib import rcParams
        rcParams['font.sans-serif'] = ['SimHei']
        rcParams['axes.unicode_minus'] = False
        self.setWindowTitle("基于基频分析的性别识别系统")
        self.setGeometry(100, 100, 1200, 900)

        # 主部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 标题
        self.title_label = QLabel("请选择音频文件或进行实时录音")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font: bold 18px; margin: 20px 0;")
        main_layout.addWidget(self.title_label)

        # 按钮区域
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        
        # 文件选择按钮
        self.file_btn = QPushButton("选择音频文件")
        self.file_btn.clicked.connect(self.load_audio)
        btn_layout.addWidget(self.file_btn)
        
        # 录音按钮
        self.record_btn = QPushButton("开始录音")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setStyleSheet("background-color: #f44336; color: white;")
        btn_layout.addWidget(self.record_btn)
        
        # 分析按钮
        self.analyze_btn = QPushButton("分析录音")
        self.analyze_btn.clicked.connect(lambda: self.analyze_audio(self.audio_data, self.RATE))
        self.analyze_btn.setEnabled(False)
        btn_layout.addWidget(self.analyze_btn)
        
        main_layout.addWidget(btn_container)

        # 结果显示
        self.result_label = QLabel("等待音频输入...")
        self.result_label.setStyleSheet("font: 14px; color: #666; margin: 20px 0;")
        main_layout.addWidget(self.result_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # 图表区域（时域+频域）
        self.figure_container = QWidget()
        figure_layout = QHBoxLayout(self.figure_container)
        
        # 时域图
        self.time_figure = Figure(figsize=(5, 4), dpi=100)
        self.time_canvas = FigureCanvas(self.time_figure)
        self.time_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        figure_layout.addWidget(self.time_canvas)
        
        # 频域图
        self.freq_figure = Figure(figsize=(5, 4), dpi=100)
        self.freq_canvas = FigureCanvas(self.freq_figure)
        self.freq_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        figure_layout.addWidget(self.freq_canvas)
        
        main_layout.addWidget(self.figure_container)

        # 实时更新定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.stream = None

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
            QPushButton {
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                font: bold 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                opacity: 0.9;
            }
        """)

    def toggle_recording(self):
        """切换录音状态"""
        # if not self.recording:
        #     # 开始录音
        #     self.recording = True
        #     self.audio_data = np.array([], dtype=np.float32)
            
        #     self.record_btn.setText("停止录音")
        #     self.record_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        #     self.result_label.setText("录音中...")
        #     self.result_label.setStyleSheet("color: #f44336; font-weight: bold;")
            
        #     # 打开音频流
        #     self.stream = self.p.open(
        #         format=self.FORMAT,
        #         channels=self.CHANNELS,
        #         rate=self.RATE,
        #         input=True,
        #         frames_per_buffer=self.CHUNK,
        #         stream_callback=self.audio_callback
        #     )
            
        #     self.timer.start(100)  # 每100ms更新一次图表
        if not self.recording:
            self.recording = True
            self.audio_data = np.array([], dtype=np.float32)
            # 显式指定输入设备索引（可选）
            # device_index = 0  # 根据输出测试脚本中的设备索引修改
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self.audio_callback,
                # input_device_index=device_index  # 如果需要指定设备
            )
            self.timer.start(100)
        else:
            # 停止录音
            self.recording = False
            self.record_btn.setText("开始录音")
            self.record_btn.setStyleSheet("background-color: #f44336; color: white;")
            self.result_label.setText("录音完成，点击分析按钮或继续录音")
            self.result_label.setStyleSheet("color: #4CAF50;")
            
            self.stream.stop_stream()
            self.stream.close()
            self.timer.stop()
            self.analyze_btn.setEnabled(True)
            
            # 保存最后5秒音频
            if len(self.audio_data) > 0:
                self.update_plots(final=True)

    def audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数，实时收集数据"""
        audio_array = np.frombuffer(in_data, dtype=np.int16)
        self.audio_data = np.concatenate((self.audio_data, audio_array.astype(np.float32) / 32768.0))
        
        # 保留最近5秒的音频数据
        max_samples = 5 * self.RATE
        if len(self.audio_data) > max_samples:
            self.audio_data = self.audio_data[-max_samples:]
        
        return (in_data, pyaudio.paContinue)

    def update_plots(self, final=False):
        """更新实时图表"""
        if len(self.audio_data) == 0:
            return
            
        # 更新时域图
        self.time_figure.clear()
        ax_time = self.time_figure.add_subplot(111)
        time_axis = np.arange(len(self.audio_data)) / self.RATE
        
        ax_time.plot(time_axis, self.audio_data, color='#4CAF50', alpha=0.8)
        ax_time.set(xlabel='时间 (秒)', ylabel='振幅', 
                   title='实时音频波形' if not final else '录音波形')
        ax_time.grid(True, linestyle='--', alpha=0.5)
        ax_time.set_xlim(max(0, time_axis[-1]-5), time_axis[-1])  # 显示最近5秒
        self.time_canvas.draw()
        
        # 更新频域图（每10次更新一次以减少计算量）
        if final or (self.timer.interval() * self.update_plots.counter % 1000 < 100):
            self.freq_figure.clear()
            ax_freq = self.freq_figure.add_subplot(111)
            
            # 计算STFT
            D = librosa.stft(self.audio_data[-self.RATE*2:])  # 使用最后2秒数据
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            librosa.display.specshow(
                S_db,
                sr=self.RATE,
                x_axis='time',
                y_axis='log',
                ax=ax_freq,
                cmap='viridis'
            )
            ax_freq.set(title='实时频谱' if not final else '录音频谱')
            self.freq_canvas.draw()
        
        self.update_plots.counter += 1
    
    update_plots.counter = 0

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
            self.audio_data = y
            self.RATE = sr
            self.analyze_audio(y, sr)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"文件加载失败：{str(e)}")

    def analyze_audio(self, y, sr):
        """分析音频并显示结果"""
        # 清除之前的绘图
        self.time_figure.clear()
        self.freq_figure.clear()

        try:
            # 基频分析
            f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=800, sr=sr)
            f0 = f0[voiced_flag]
            
            if len(f0) == 0:
                self.show_result("无法检测到有效基频", "#ff0000")
                return

            avg_f0 = np.nanmean(f0)
            gender = "女性" if avg_f0 > 165 else "男性"
            confidence = min(100, max(0, abs(avg_f0 - 165) / 1.65))  # 简单置信度计算
            self.show_result(
                f"基频平均值：{avg_f0:.1f}Hz → {gender} (置信度: {confidence:.0f}%)", 
                "#2196F3" if gender == "男性" else "#E91E63"
            )

            # 绘制完整时域图
            ax_time = self.time_figure.add_subplot(111)
            time = np.arange(len(y)) / sr
            ax_time.plot(time, y, color='#4CAF50', alpha=0.8)
            ax_time.set(xlabel='时间 (秒)', ylabel='振幅', title='音频波形 (时域)')
            ax_time.grid(True, linestyle='--', alpha=0.5)
            self.time_canvas.draw()

            # 绘制完整频域图
            ax_freq = self.freq_figure.add_subplot(111)
            D = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            img = librosa.display.specshow(
                S_db,
                sr=sr,
                x_axis='time',
                y_axis='log',
                ax=ax_freq,
                cmap='viridis'
            )
            self.freq_figure.colorbar(img, ax=ax_freq, format="%+2.0f dB")
            ax_freq.set(title='频谱图 (对数频率轴)')
            self.freq_canvas.draw()

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

    def closeEvent(self, event):
        """关闭窗口时释放资源"""
        if self.recording:
            self.toggle_recording()
        if hasattr(self, 'p'):
            self.p.terminate()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GenderRecognitionApp()
    window.show()
    sys.exit(app.exec())