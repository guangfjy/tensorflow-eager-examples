import os
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
# make_interp_spline(spline已过时) 函数主要帮助我们实现简单的平滑化处理
from scipy.interpolate import make_interp_spline
plt.switch_backend('agg')  # 防止ssh上绘图问题


class LrFinder():
    def __init__(self, model, data_size, batch_szie, fig_path):
        '''
        ---
        model: 我们需要训练的模型
        data_size: 数据集大小
        batch_size: 批大小
        fig_path: 保存图片的路径
        ---
        '''
        # 进度条变量
        self.data_size = data_size
        self.batch_size = batch_szie
        self.loss_name = 'loss'
        self.eval_name = 'acc'
        self.width = 30
        self.resiud = self.data_size % self.batch_size
        self.n_batch = self.data_size // self.batch_size

        # lr_find变量
        self.model = model
        self.best_loss = 1e9
        self.fig_path = fig_path
        self.losses = []
        self.lrs = []

        self.model_status = False  # 查找结束的标志
        # 初始化了一个learning_rate变量，在Tensorflow中，我们将对learning_rate进行更新
        self.learning_rate = tf.Variable(0.001, trainable=False)

    def on_batch_end(self, loss,):
        '''
        每次batch更新进行操作
        on_batch_end函数名表明了，我们应该在一个batch结束之后，运行该函数
        '''
        # 更新学习率和loss列表数据，将根据该列表进行绘制图标数据
        self.lrs.append(self.learning_rate.numpy())
        self.losses.append(loss.numpy())

        # 对loss进行判断，主要当loss增加时，停止训练（该规则主要是从fastai模块中获取）
        if loss.numpy() > self.best_loss * 4:
            self.model_status = True

        #　更新best_loss
        if loss.numpy() < self.best_loss:
            self.best_loss = loss.numpy()

        # 学习率更新方式
        lr = self.lr_mult * self.learning_rate.numpy()  # self.lr_mult是更新因子
        self.learning_rate.assign(lr)  # 在Tensorflow中对变量重新赋值主要是使用assign函数。

    def progressbar(self, batch, loss, acc):
        # 自定义的进度条信息
        recv_per = int(100 * (batch + 1) / self.n_batch)
        if batch == self.n_batch:
            num = batch * self.batch_size + self.resiud
        else:
            num = batch * self.batch_size
        if recv_per >= 100:
            recv_per = 100
        # 字符串拼接的嵌套使用
        show_bar = ('[%%-%ds]' % self.width) % (int(self.width * recv_per / 100) * ">")
        show_str = '\r%d/%d %s - %s: %.4f- %s: %.4f'
        print(show_str % (
            num, self.data_size,
            show_bar,
            self.loss_name, loss,
            self.eval_name, acc),
            end='')

    def plot_loss(self):
        # 对loss进行可视化，主要是对每个batch的loss进行绘制图表
        plt.style.use("ggplot")
        plt.figure()
        plt.ylabel("loss")
        plt.xlabel("learning rate")
        plt.plot(self.lrs, self.losses)
        plt.xscale('log')
        plt.savefig(os.path.join(self.fig_path, 'loss.jpg'))
        plt.close()

    def plot_loss_smooth(self):
        # 这里我们使用scipy模块简单点进行平滑化，方便查看

        # 这里我们采用移动平均进行平滑化
        xnew = np.linspace(min(self.lrs), max(self.lrs), 100)
        spl = make_interp_spline(self.lrs, self.losses, k=3)  # 使用scipy模块中的make_interp_spline函数简单实现一个平滑的效果
        smooth_loss = spl(xnew)
        plt.ylabel("rate of loss change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(xnew, smooth_loss)
        plt.xscale('log')
        plt.savefig(os.path.join(self.fig_path, 'smooth_loss.jpg'))
        plt.close()

    def find(self, trainDataset, start_lr, end_lr, optimizer, epochs=1, verbose=1, save=True):
        '''
        定义一个拟合函数，对数据集进行训练并保存loss和lr变化
        ---
        trainDataset: 训练数据集，这里是Dataset格式
        start_lr: 开始学习率，一般设置为0.000001
        end_lr: 结束学习率，一般设置为10
        optimizer: 优化器
        epochs: 训练的epoch总数，在这里一般设置为１
        verbose: 是否打印信息，默认为1
        save: 是否保存图像
        ---
        '''
        self.learning_rate.assign(start_lr)

        num_batches = epochs * self.data_size / self.batch_size

        # 更新因子
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))

        for i in range(epochs):
            print("\nEpoch {i}/{epochs}".format(i=i + 1, epochs=epochs))
            for (batch_id, (X, y)) in enumerate(trainDataset):
                y_pred, train_loss, grads = self.model.compute_grads(X, y)
                train_score = self.model.compute_score(y_pred=y_pred, y_true=y)
                optimizer.apply_gradients(zip(grads, self.model.variables))

                self.on_batch_end(loss=train_loss)
                if verbose > 0:
                    self.progressbar(batch=batch_id, loss=train_loss, acc=train_score)
                if self.model_status:
                    break

            if self.model_status:
                break
        if save:
            self.plot_loss()
            self.plot_loss_smooth()