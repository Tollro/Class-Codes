import sounddevice as sd
import numpy as np
import librosa
import time
import os
from scipy.io.wavfile import write
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.signal import butter, filtfilt, lfilter

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class VoiceAnalyzer:
    @staticmethod
    def apply_bandpass_filter(signal, sr, lowcut=80, highcut=300, order=4):
        """
        应用带通滤波器以滤除谐波
        :param signal: 输入信号
        :param sr: 采样率
        :param lowcut: 低截止频率
        :param highcut: 高截止频率
        :param order: 滤波器阶数
        :return: 滤波后的信号
        """
        # 计算归一化频率
        nyquist = 0.5 * sr
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # 设计Butterworth带通滤波器
        b, a = butter(order, [low, high], btype='band')
        
        # 应用前向-后向滤波以消除相位失真
        filtered_signal = filtfilt(b, a, signal)
        
        return filtered_signal

    @staticmethod
    def extract_features(signal, sr):
        """提取声学特征"""
        features = {}

        # 应用带通滤波器滤除谐波
        filtered_signal = VoiceAnalyzer.apply_bandpass_filter(signal, sr)

        # 使用librosa提取基频，f0储存了每个帧对应的估计值
        f0, voiced_flag, voiced_probs = librosa.pyin(signal, fmin=80, fmax=300, sr=sr)
        
        valid_f0 = f0[~np.isnan(f0)]
        # 平均值
        f0_average = np.average(valid_f0)
        features['f0_average'] = f0_average
        # 中位数（仅考虑有声部分）
        f0_median = np.nanmedian(valid_f0)
        features['f0_median'] = f0_median
        # 确保平均值在合理范围内
        if 80 < f0_average < 300:
            features['f0'] = f0_median
        else:
            messagebox.showinfo("提示", "未识别到有效基频值")
            features['f0'] = None
            
        return features, filtered_signal

class GenderRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("语音性别识别系统")
        self.root.geometry("1200x900")
        
        # 初始化参数
        self.sample_rate = 44100
        self.duration = tk.IntVar(value=3)
        self.file_path = tk.StringVar()
        self.recording = None
        self.result_text = tk.StringVar(value="等待操作...")

        # 滤波器参数
        self.filter_lowcut = tk.IntVar(value=80)
        self.filter_highcut = tk.IntVar(value=300)
        self.filter_order = tk.IntVar(value=4)

        # 创建界面
        self.create_widgets()
        self.create_visualization()
        
    def create_visualization(self):
        """创建可视化图表"""
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.axs = {
            'original_waveform': self.figure.add_subplot(221),
            'original_spectrum': self.figure.add_subplot(222),
            'filtered_waveform': self.figure.add_subplot(223),
            'filtered_spectrum': self.figure.add_subplot(224)
        }
        
        # 设置图表样式
        self.axs['original_waveform'].set_title("原始信号 - 时域波形")
        self.axs['original_waveform'].set_xlabel("时间（秒）")
        self.axs['original_waveform'].set_ylabel("幅度")
        
        self.axs['original_spectrum'].set_title("原始信号 - 频域频谱")
        self.axs['original_spectrum'].set_xlabel("频率（Hz）")
        self.axs['original_spectrum'].set_ylabel("幅度")
        
        self.axs['filtered_waveform'].set_title("滤波后信号 - 时域波形")
        self.axs['filtered_waveform'].set_xlabel("时间（秒）")
        self.axs['filtered_waveform'].set_ylabel("幅度")
        
        self.axs['filtered_spectrum'].set_title("滤波后信号 - 频域频谱")
        self.axs['filtered_spectrum'].set_xlabel("频率（Hz）")
        self.axs['filtered_spectrum'].set_ylabel("幅度")

        self.figure.subplots_adjust(hspace=0.5, wspace=0.3)  # 调整间距
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    
    def create_widgets(self):
        """创建界面控件"""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10, fill="x")
        
        # 录音控制
        record_frame = ttk.LabelFrame(control_frame, text="录音控制")
        record_frame.pack(side=tk.LEFT, padx=10, fill="both", expand=True)
        
        ttk.Label(record_frame, text="录音时长（秒）:").grid(row=0, column=0)
        ttk.Entry(record_frame, textvariable=self.duration, width=5).grid(row=0, column=1)
        ttk.Button(record_frame, text="🎤 开始录音", command=self.start_recording).grid(row=0, column=2, padx=5)
        ttk.Button(record_frame, text="📊 分析录音", command=self.analyze_recording).grid(row=0, column=3)
        
        # 文件操作
        file_frame = ttk.LabelFrame(control_frame, text="文件操作")
        file_frame.pack(side=tk.LEFT, padx=10, fill="both", expand=True)
        
        ttk.Entry(file_frame, textvariable=self.file_path, width=40).grid(row=0, column=0)
        ttk.Button(file_frame, text="📂 浏览文件", command=self.browse_file).grid(row=0, column=1)
        ttk.Button(file_frame, text="🔍 分析文件", command=self.analyze_file).grid(row=0, column=2)

        # 滤波器设置
        filter_frame = ttk.LabelFrame(control_frame, text="滤波器设置")
        filter_frame.pack(side=tk.LEFT, padx=10, fill="both", expand=True)
        
        ttk.Label(filter_frame, text="低截止频率 (Hz):").grid(row=0, column=0)
        ttk.Entry(filter_frame, textvariable=self.filter_lowcut, width=5).grid(row=0, column=1)
        
        ttk.Label(filter_frame, text="高截止频率 (Hz):").grid(row=0, column=2)
        ttk.Entry(filter_frame, textvariable=self.filter_highcut, width=5).grid(row=0, column=3)
        
        ttk.Label(filter_frame, text="滤波器阶数:").grid(row=0, column=4)
        ttk.Entry(filter_frame, textvariable=self.filter_order, width=3).grid(row=0, column=5)

        # 结果显示
        result_frame = ttk.LabelFrame(self.root, text="分析结果")
        result_frame.pack(pady=10, fill="both", expand=True)
        
        self.result_label = ttk.Label(
            result_frame, 
            textvariable=self.result_text,
            wraplength=1100,
            font=('微软雅黑', 12),
            justify="left"
        )
        self.result_label.pack(pady=10, padx=10, fill="both", expand=True)
        
        # 状态栏
        self.status_bar = ttk.Label(self.root, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def start_recording(self):
        """开始录音"""
        try:
            duration = self.duration.get()
            if duration <= 0:
                messagebox.showerror("错误", "录音时长必须大于0")
                return
            self.update_status(f"正在录音...剩余时间 {duration} 秒")
            self.recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            self.root.after(100, self.update_recording_timer, duration)
        except Exception as e:
            messagebox.showerror("录音错误", str(e))
    
    def update_recording_timer(self, remaining):
        """更新录音倒计时"""
        if remaining > 0:
            self.update_status(f"录音中...剩余时间 {remaining} 秒")
            self.root.after(1000, self.update_recording_timer, remaining-1)
        else:
            sd.wait()
            self.recording = self.recording.flatten()
            self.update_status("录音完成")
            messagebox.showinfo("提示", "录音已就绪，点击分析按钮查看结果")
    
    def analyze_recording(self):
        """分析录音"""
        if self.recording is None:
            messagebox.showwarning("警告", "请先进行录音")
            return
        try:
            self.analyze_signal(self.recording, self.sample_rate, "录音文件")
        except Exception as e:
            messagebox.showerror("分析错误", str(e))
    
    def browse_file(self):
        """浏览文件"""
        filetypes = [("音频文件", "*.wav *.mp3"), ("所有文件", "*.*")]
        filename = filedialog.askopenfilename(title="选择音频文件", filetypes=filetypes)
        if filename:
            self.file_path.set(filename)
            self.update_status(f"已选择文件: {filename}")
    
    def analyze_file(self):
        """分析文件"""
        filepath = self.file_path.get()
        if not filepath:
            messagebox.showwarning("警告", "请先选择音频文件")
            return
        try:
            signal, sr = librosa.load(filepath, sr=None, mono=True)
            self.analyze_signal(signal, sr, os.path.basename(filepath))
        except Exception as e:
            messagebox.showerror("分析错误", str(e))
    
    def analyze_signal(self, signal, sr, source):
        """核心分析流程"""
        try:
            # 确保信号是浮点类型
            signal = np.asarray(signal, dtype=np.float32)
            
            # 清空图表
            for ax in self.axs.values():
                ax.clear()
            
            # 可视化1：时域波形
            time_axis = np.arange(len(signal)) / sr
            self.axs['original_waveform'].plot(time_axis, signal)
            self.axs['original_waveform'].set_title("时域波形")
            self.axs['original_waveform'].set_xlabel("时间（秒）")
            self.axs['original_waveform'].set_ylabel("幅度（归一化）")
            self.axs['original_waveform'].set_xlim(0, time_axis[-1])  # 完整显示时间范围
            
            # 可视化2：频域频谱
            fft = np.abs(np.fft.rfft(signal))
            fft_db = 20 * np.log10(fft + 1e-6)  # 转换为dB
            freqs = np.fft.rfftfreq(len(signal), 1/sr)
            self.axs['original_spectrum'].plot(freqs, fft_db)
            self.axs['original_spectrum'].set_title("频域频谱")
            self.axs['original_spectrum'].set_xlabel("频率（Hz）")
            self.axs['original_spectrum'].set_ylabel("幅度（dB）(相对值)")

           # 标注关键频段
            self.axs['original_spectrum'].axvspan(80, 300, alpha=0.2, color='yellow')  # 人声基频区
            self.axs['original_spectrum'].set_xlim(0, 500)  # 聚焦人声主要频段
            
            # 提取特征
            features, filtered_signal = VoiceAnalyzer.extract_features(signal, sr)

            # 可视化3：滤波后信号时域波形
            self.axs['filtered_waveform'].plot(time_axis, filtered_signal)
            self.axs['filtered_waveform'].set_title("滤波后信号 - 时域波形")
            self.axs['filtered_waveform'].set_xlabel("时间（秒）")
            self.axs['filtered_waveform'].set_ylabel("幅度")
            self.axs['filtered_waveform'].set_xlim(0, time_axis[-1])

            # 可视化4：滤波后信号频域频谱
            fft_filtered = np.abs(np.fft.rfft(filtered_signal))
            fft_filtered_db = 20 * np.log10(fft_filtered + 1e-6)
            self.axs['filtered_spectrum'].plot(freqs, fft_filtered_db)
            self.axs['filtered_spectrum'].set_title("滤波后信号 - 频域频谱")
            self.axs['filtered_spectrum'].set_xlabel("频率（Hz）")
            self.axs['filtered_spectrum'].set_ylabel("幅度 (dB)")

            self.axs['filtered_spectrum'].axvspan(80, 300, alpha=0.2, color='yellow')  # 人声基频区
            self.axs['filtered_spectrum'].set_xlim(0, 500)  # 聚焦人声主要频段

            # 查找（在 80~300Hz 范围内找能量最大值）
            fmin, fmax = 80, 300
            mask = (freqs >= fmin) & (freqs <= fmax)
            if np.any(mask):
                peak_index_in_mask = np.argmax(fft_filtered_db[mask])
                original_index = np.where(mask)[0][peak_index_in_mask]
                peak_freq = freqs[original_index]
                peak_db = fft_filtered_db[original_index]
            else:
                peak_freq = np.nan
                peak_db = np.nan

            features['f_peak'] = peak_freq

            # 标注
            if features['f_peak']:
                self.axs['filtered_spectrum'].axvline(features['f_peak'], color='r', linestyle='--')
                self.axs['filtered_spectrum'].text(
                    features['f_peak'], np.max(fft_db)-10,
                    f"峰值: {features['f_peak']:.1f}Hz",
                    color='r', ha='center'
                )
            # 标注
            if features['f0']:
                self.axs['filtered_spectrum'].axvline(features['f0'], color='b', linestyle='--')
                self.axs['filtered_spectrum'].text(
                    features['f0'], np.max(fft_db)-10,
                    f"中位数: {features['f0']:.1f}Hz",
                    color='b', ha='center'
                )


            # 判断性别
            if features['f0'] < 160:
                gender = "男性"
            else:
                gender = "女性"
            
            # 构建结果报告
            result = f"""=== 语音分析报告 ===
数据来源：{source}
采样率：{sr} Hz
-------------------------
【关键特征】
峰值：{features['f_peak']:.1f} Hz
yin中位数：{features['f0']:.1f} Hz
yin平均数：{features['f0_average']:.1f} Hz
-------------------------
【判断结果】
性别预测：{gender}"""
            
            self.result_text.set(result)
            self.update_status("分析完成")
            
            # 重绘图表
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("分析错误", str(e))
    
    def update_status(self, message):
        """更新状态栏"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = GenderRecognitionApp(root)
    root.mainloop()