# 导入系统模块（用于处理退出等操作）
import sys
# 导入数值计算库（用于生成信号数据）
import numpy as np
# 从PyQt6的部件库导入必要组件
from PyQt6.QtWidgets import (
    QApplication,   # 应用主类
    QMainWindow,    # 主窗口基类
    QWidget,        # 基础部件
    QVBoxLayout,    # 垂直布局
    QHBoxLayout,    # 水平布局
    QPushButton,    # 按钮部件
    QLabel,         # 文本显示控件
    QLineEdit,      # 单行文本编辑框控件
    QMessageBox     # 消息框控件
)
# 导入Matplotlib的Qt后端支持
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # 画布组件
from matplotlib.figure import Figure  # 绘图画布

class DiscreteSignalApp(QMainWindow):
    def __init__(self): #类的构造函数，在创建对象时自动调用
        super().__init__()  # 用父类的构造函数，确保正确初始化
        #设置字体
        from matplotlib import rcParams
        rcParams['font.sans-serif'] = ['SimHei']
        rcParams['axes.unicode_minus'] = False
        # 窗口基本设置
        self.setWindowTitle("离散信号演示")  # 设置窗口标题
        self.setGeometry(150, 150, 850, 650)       # 设置窗口位置和大小(x, y, width, height)
        
        #           创建主界面布局结构             #
        # 创建中央部件容器
        main_widget = QWidget()
        self.setCentralWidget(main_widget)  # 设置主窗口的中央部件
        # 创建垂直布局管理器（主布局）
        main_layout = QVBoxLayout(main_widget)  # 将布局绑定到main_widget
        # 创建水平布局用于放置按钮
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)  # 将按钮布局添加到主布局顶部
        # 新增周期输入组件
        self.period_layout = QHBoxLayout()
        main_layout.addLayout(self.period_layout)
        self.period_label = QLabel("周期 (T):")
        self.period_layout.addWidget(self.period_label)
        self.period_input = QLineEdit()
        self.period_input.setPlaceholderText("输入正整数")
        self.period_input.setText("4")  # 默认值
        self.period_layout.addWidget(self.period_input)

        #         创建Matplotlib绘图组件           #
        # 创建图形对象（8英寸宽×4英寸高，100DPI分辨率）
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)  # 添加子图（1行1列第1个）
        # 创建Qt兼容的画布组件
        self.canvas = FigureCanvasQTAgg(self.figure)
        main_layout.addWidget(self.canvas)  # 将画布添加到主布局底部
        #             创建功能按钮                #
        self.create_buttons(button_layout)  # 调用按钮创建方法

        # 初始化图形设置
        self.ax.grid(True)     # 启用网格线
        self.canvas.draw()     # 初始绘制空图形

    def create_buttons(self, layout):
        """创建按钮并绑定事件处理"""
        # 按钮配置列表：（显示文本，点击处理函数）
        signals = [
            ("单位脉冲信号", self.plot_unit_impulse),
            ("单位阶跃离散信号", self.plot_unit_step),
            ("指数离散信号", self.plot_exponential),
            ("正弦连续信号", self.plot_sinusoidal)
        ]
        
        # 循环创建按钮//链接对应处理函数
        for text, handler in signals:
            btn = QPushButton(text)            # 创建按钮实例
            btn.clicked.connect(handler)       # 绑定点击信号到处理函数
            layout.addWidget(btn)              # 将按钮添加到布局中

    def clear_plot(self):
        """清空当前绘图"""
        self.ax.clear()     # 清除坐标系内容
        self.ax.grid(True)   # 重新启用网格线

    def plot_unit_impulse(self):
        """绘制单位脉冲信号"""
        self.clear_plot()  # 清空画布
        n = np.arange(-8, 8)  # 生成-5到5的整数序列（共11个点）
        y = np.where(n == 0, 1, 0)  # 创建脉冲信号（n=0时为1，其他为0）
        
        # 绘制离散信号（stem表示火柴棒图）
        self.ax.stem(n, y, basefmt=" ")  # basefmt设置基线格式
        self.ax.set_title("delta[n]")  # 设置标题
        self.canvas.draw()  # 更新画布显示

    def plot_unit_step(self):
        """绘制单位阶跃离散信号"""
        self.clear_plot()
        n = np.arange(-8, 8)
        y = np.where(n >= 0, 1, 0)  # n≥0时为1，否则为0
        self.ax.stem(n, y, basefmt=" ")
        self.ax.set_title("u[n]")
        self.canvas.draw()

    def plot_exponential(self):
        """绘制指数衰减离散信号"""
        self.clear_plot()
        n = np.arange(-8, 8)
        y = 0.5**n * (n >= 0)  # 0.5的n次方（仅当n≥0时有效）
        self.ax.stem(n, y, basefmt=" ")
        self.ax.set_title("指数信号 (0.5^n)")
        self.canvas.draw()

    def plot_sinusoidal(self):
        self.clear_plot()
        try:
            # 获取并验证输入
            T = float(self.period_input.text())
            if T <= 0 or not T.is_integer():
                raise ValueError("必须是正整数")
            T = int(T)
        except ValueError:
            QMessageBox.critical(self, "错误", "请输入有效的正整数周期值")
            return

        # 根据周期生成信号
        n = np.arange(-2*T, 2*T+1)  # 显示两个完整周期
        omega = 2 * np.pi / T      # 角频率计算
        y = np.sin(omega * n)
        
        # 绘制图形
        self.ax.stem(n, y, basefmt=" ", linefmt='C0-', markerfmt='C0o')
        self.ax.set_title(f"正弦信号 (周期={T})")
        self.canvas.draw()

    # def plot_sinusoidal(self):
    #     """绘制正弦连续信号"""
    #     self.clear_plot()
    #     x1 = np.arange(-8, 8)
    #     sgl_sin = np.sin(x1)
    #     self.ax.stem(x1,sgl_sin)
    #     self.ax.set_title("正弦离散信号 sin(t)")
    #     self.canvas.draw()

if __name__ == "__main__":
    # 创建Qt应用实例（必须）
    app = QApplication(sys.argv)  
    # 创建主窗口实例
    window = DiscreteSignalApp()
    # 显示窗口
    window.show()
    # 启动应用事件循环，sys.exit确保程序退出时返回状态码
    sys.exit(app.exec())