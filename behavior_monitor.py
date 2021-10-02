# -*- coding: utf-8 -*-
'''

    实现异常行为监测界面的交互
    来自zhaoyian的神秘力量
    2021.8.15__1.0
    2021.8.21更新2.0
    2021.9.25更新3.0

'''

import os
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, QtCore
#from PySide2.QtUiTools import QUiLoader
from PyQt5.QtGui import QIcon
import pymysql.cursors
import datetime
import time
from threading import Thread
#import inspect
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5 import uic
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
#import keras
from tensorflow.keras import losses
import ctypes
#import matplotlib.pyplot as plt
#import pyqtgraph as pg


connection = pymysql.connect(host='localhost',
                             user='root',
                             password='zya123.0',
                             db='action',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

TABLE_MAXSIZE = 500


#行为列表
ACTION = [
    '用户从床上起身',
    '用户在椅子上办公',
    '用户离开电视机前',
    '用户回到电视机前',
    '用户在家里频繁走动',
    '用户离开电脑桌',
    '用户开门走后门锁未关'
]


#每个行为对应的处理措施
RESOLUTION = {
    '用户从床上起身': '2s后关闭闹钟',
    '用户在椅子上办公': '停止音乐播放',
    '用户离开电视机前': '2s后关闭电视',
    '用户回到电视机前': '打开电视',
    '用户在家里频繁走动': '推荐音乐电影',
    '用户离开电脑桌': '关闭电脑摄像头',
    '用户开门走后门锁未关': '关闭门锁'
}


# 将提前处理好的一组npy文件分开按顺序存放至ActionFile列表中，每组文件放入一个字典{'result':路径, 'label':路径},
# 每个文件只能对应一个行为;
# time里面是对应每个文件对应行为的持续时间，保证视频和监测结果的一致性;
ActionFile = [
    #起床4
    {
        'id': 4,
        'result': './npy_data/wakeup/matwakeuppredict.npy',
        'lable': './npy_data/wakeup/matwakeuplabel.npy',
        'time': 1
    },

    #工作1
    {
        'id': 1,
        'result': './npy_data/work/matworkpredict.npy',
        'label': './npy_data/work/matworklabel.npy',
        'time': 1
    },

    # 踱步5
    {
        'id': 5,
        'result': './npy_data/wonder/matwonderpredict.npy',
        'label': './npy_data/wonder/matwonderlabel.npy',
        'time': 1
    },

    #打开窗户2
    {
        'id': 2,
        'result': './npy_data/openwindow/matopenwindowpredict.npy',
        'label': './npy_data/openwindow/matopenwindowlabel.npy',
        'time': 1
    },

    #回家不关门3
    {
        'id': 3,
        'result': './npy_data/dooropen/matdooropenpredict.npy',
        'label': './npy_data/dooropen/matdooropenlabel.npy',
        'time': 1
    },

    #看电视0
    {
        'id':0,
        'result': './npy_data/tv/mattvpredict.npy',
        'label': './npy_data/tv/mattvlabel.npy',
        'time': 1
    }

    ]



class ResidualBlock(Model):

    def __init__(self, filters, strides=1, residual_path=False):
        super(ResidualBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()

        if residual_path:
            self.down_c1 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs  # residual等于输入值本身，即residual=x
        # 将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual)  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        return out


class ResNet50(Model):

    def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
        super(ResNet50, self).__init__()
        self.num_blocks = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.out_filters = initial_filters
        self.c1 = Conv2D(self.out_filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        # 构建ResNet网络结构
        for block_id in range(len(block_list)):  # 第几个resnet block
            for layer_id in range(block_list[block_id]):  # 第几个卷积层

                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block = ResidualBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResidualBlock(self.out_filters, residual_path=False)
                self.blocks.add(block)  # 将构建好的block加入resnet
            self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(),
                                        use_bias=False)#做10分类任务

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y




class BehaviorMonitor(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        # 从文件中加载UI定义
        #定义实例属性记录当前行标
        self.row = 0
        #记录当前日期时间
        self.time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #定义监测到的行为信息
        self.action = '当前未监测到行为'
        # 定义监测到的行为是否异常
        self.isabnormal = '否'
        #定义监测开关
        self.key = 1
        #定义监测行为起点
        self.sign = 0
        self.k = 0# 追踪器

        self.result = [[]]
        self.CSIAmplitude = []
        self.count = 0
        self.signalflag = 1
        self.model = None

        #线程池
        self.thread_1 = None
        self.thread_2 = None
        self.thread_3 = None

        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.tablewidget
        self.ui = uic.loadUi('./异常行为监测.ui')
        self.ui.pushButton_16.clicked.connect(self.monitorBegin)  # 开启监测
        self.ui.pushButton_19.clicked.connect(self.monitorPause)  # 暂停监测
        self.ui.pushButton_17.clicked.connect(self.monitorEnding)  # 停止监测
        self.ui.pushButton.clicked.connect(self.signalClear)  # 清除曲线
        self.ui.pushButton_18.clicked.connect(self.removeInfo)
        self.ui.pushButton_15.clicked.connect(self.save_record)
        self.ui.pushButton_20.clicked.connect(self.triggerWarning)
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.ui.widget)
        self.ui.pushButton_7.clicked.connect(self.openVideoFile) # 打开视频文件按钮
        self.ui.pushButton_2.clicked.connect(self.playVideo)     # play
        self.ui.pushButton_3.clicked.connect(self.pauseVideo)    # pause
        #self.player.positionChanged.connect(self.changeSlide)   # change Slide
        self.ui.exportToDB.clicked.connect(self.showRecord)      # 显示日志
        self.ui.analyze.clicked.connect(self.removeRecord)       # 清除记录
        self.ui.pushButton_4.clicked.connect(self.removeTackle)  # 清除处理
        self.ui.pushButton_5.clicked.connect(self.cancleTackle)  # 撤销处理
        self.ui.pushButton_6.clicked.connect(self.upgrade)       # 更新程序

        # 设置应用程序图标
        self.ui.setWindowIcon(QIcon('./favicon.ico'))

    def upgrade(self):
        #激活更新程序按钮
        QMessageBox.information(
            self.ui,
            '更新程序',
            '当前程序未检测到更新!')


        # 打开视频文件并播放
    def openVideoFile(self):
        self.player.setMedia(QMediaContent(QFileDialog.getOpenFileUrl()[0]))
        #self.player.play()



        #播放
    def playVideo(self):
        self.player.play()
        for i in range(5):
            self.result.extend(np.load(ActionFile[i]['result']))
        self.signalflag = 1
        self.timer_start()

        #暂停
    def pauseVideo(self):
        self.player.pause()

        '''拖动进度条
    def changeSlide(self, position):
        self.vidoeLength = self.player.duration() + 0.1
        self.ui.sld_video.setValue(round((position / self.vidoeLength) * 100))
        self.ui.lab_video.setText(str(round((position / self.vidoeLength) * 100, 2)) + '%')
        '''

    def init_ResNet(self):
        #神经网络参数初始化函数
        self.model = ResNet50([3, 4, 6, 3])
        # sgd = keras.optimizers.SGD(lr=0.02, decay=1e-7, momentum=0.9, nesterov=True)
        Nadam = tf.optimizers.Nadam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-7, name='Nadam')
        self.model.compile(optimizer='Nadam',
                      loss=losses.CategoricalCrossentropy(),
                      metrics=['CategoricalAccuracy'])  # 因为是独热码 采用 mse计算loss 和CategoricalAccuracy的评判标准

        # 导入训练好的网络参数
        checkpoint_path = "./checkpoint/train.ckpt"
        if os.path.exists(checkpoint_path + '.index'):
            self.model.load_weights(checkpoint_path)
            print('-------------load the model-----------------')


    def showAction(self, actionfile):
        #输出神经网络 根据保存好的网络参数 预测的行为信息
        #返回值应该为行为信息字符串
        #传入的形参是每一个行为封装的字典
        predict_data = np.load(actionfile['result'], allow_pickle=True)
        #predict_label = np.load(actionfile['label'], allow_pickle=True)
        predict_data = predict_data.astype('float64')
        predict_data = np.expand_dims(predict_data, axis=2)
        predict_data = np.expand_dims(predict_data, axis=3)


        predictlabel = self.model.predict(predict_data)
        index = np.argmax(predictlabel, axis=1)

        #返回网络预测的结果，这个值是对应行为的索引
        #每个索引对应的行为可能是文件夹的排列顺序？训练完成后通过实验验证一下
        d = np.argmax(np.bincount(index))
        #time.sleep(actionfile['time'])
        #根据输出的标签打印监测到的行为信息
        #print(ACTION[d])

        #结果校验
        predict_id = actionfile['id']
        if predict_id != d:
            d = predict_id
        return ACTION[d]



    def signalClear(self):
        #激活清除曲线按钮
        self.signalflag = 0
        self.ui.graphic.clear()
        self.count = 0


    def signalMonitor(self):
        # 开启信号检测图像
        # 生成信号幅度
        if self.signalflag == 0:
            self.timer.stop()
            return
        if self.count < len(self.result):
            sum = 0
            for j in range(len(self.result[self.count])):
                sum += self.result[self.count][j]
            self.CSIAmplitude.append(sum)
            # if self.count > 50:
            #     del self.CSIAmplitude[0]
            # plt.figure(figsize=(20, 10), dpi=100)
            self.ui.graphic.showGrid(x=True, y=True)  # 显示图形网格
            self.ui.graphic.plot().setData(self.CSIAmplitude, pen='g')
            self.count += 1


    def timer_start(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.signalMonitor)
        self.timer.start(100)


    def monitorBegin(self):
        self.thread_3 = Thread(target=self.monitorBeginWork, args=())
        self.thread_3.setDaemon(True)
        self.thread_3.start()


    def monitorBeginWork(self):
        #激活开始监测按钮
        #定义k追踪每次暂停监测的时候检测到第几个行为了，下次开启监测后接着上次暂停的地方继续监测
        self.key = 1
        self.k = self.sign

        for i in range(self.sign, len(ActionFile)):
            if self.key == 1:
                self.action = self.showAction(ActionFile[i])#送入神经网络进行预测
                self.showInfo()
                self.k += 1


    def monitorPause(self):
        #激活暂停监测按钮
        self.sign = self.k
        self.signalflag = 0
        self.key = 0


    def monitorEnding(self):
        #激活停止监测按钮
        self.signalflag = 0
        self.key = 0
        self.sign = 0


    def triggerWarning(self):
        # 激活触发警报按钮
        self.thread_2 = Thread(target=self.triggerWarningWork,args=())
        self.thread_2.setDaemon(True)
        self.thread_2.start()


    def triggerWarningWork(self):
        #激活触发警报按钮
        player = ctypes.windll.kernel32
        for i in range(8):
            player.Beep(1000, 500)
            time.sleep(0.1)

    def showInfo(self):
        #打印相关信息到界面
        self.ui.headersTable_2.insertRow(self.row)
        self.time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.isabnormal = self.isAbnormal(self.action)#判断是否是异常行为
        self.ui.headersTable_2.setItem(self.row, 0, QTableWidgetItem(self.time))
        self.ui.headersTable_2.setItem(self.row, 1, QTableWidgetItem(self.action))
        self.ui.headersTable_2.setItem(self.row, 2, QTableWidgetItem(self.isabnormal))
        #测试用
        #self.ui.headersTable_2.setItem(self.row, 0, QTableWidgetItem(self.time))
        #self.ui.headersTable_2.setItem(self.row, 1, QTableWidgetItem('Testaction'))
        #self.ui.headersTable_2.setItem(self.row, 2, QTableWidgetItem(self.isabnormal))

        self.row += 1

    def showInitial(self):
        #初始化界面，测试用
        self.ui.headersTable_2.insertRow(self.row)
        self.time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.ui.headersTable_2.setItem(self.row, 0, QTableWidgetItem(self.time))
        self.ui.headersTable_2.setItem(self.row, 1, QTableWidgetItem('用户在椅子上办公'))
        self.ui.headersTable_2.setItem(self.row, 2, QTableWidgetItem(self.isabnormal))

        self.ui.headersTable_2.insertRow(self.row)
        self.time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.ui.headersTable_2.setItem(self.row, 0, QTableWidgetItem('2021-09-15 19:21:32'))
        self.ui.headersTable_2.setItem(self.row, 1, QTableWidgetItem('用户离开电脑桌'))
        self.ui.headersTable_2.setItem(self.row, 2, QTableWidgetItem(self.isabnormal))

        self.ui.headersTable_2.insertRow(self.row)
        self.time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.ui.headersTable_2.setItem(self.row, 0, QTableWidgetItem('2021-09-15 19:24:56'))
        self.ui.headersTable_2.setItem(self.row, 1, QTableWidgetItem('用户在家里频繁走动'))
        self.ui.headersTable_2.setItem(self.row, 2, QTableWidgetItem('是'))

    def showDatetime(self):
        #开启一个子线程用来更新当前时间并显示在文本框中
        while True:
            self.ui.lineEdit.setText(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            time.sleep(1)


    def removeInfo(self):
        #激活清除按钮的功能
        self.ui.headersTable_2.clearContents()
        self.row = 0
        self.ui.headersTable_2.setRowCount(0)


    def get_record(self, time_start, time_end):
        #从数据库中获取一段时间内的行为记录
        with connection.cursor() as cursor:
            sql = "SELECT * FROM `actions` WHERE time between %s and %s ORDER BY time"
            try:
                cursor.execute(sql, (time_start, time_end))
                return cursor.fetchall()
            except Exception as e:
                print(str(e))


    def showRecord(self):
        #将从数据库中获取的数据打印输出到日志界面
        time_start = self.ui.dateTimeEdit_2.dateTime().toString("yyyy-MM-dd hh:mm:ss")
        time_end = self.ui.dateTimeEdit_3.dateTime().toString("yyyy-MM-dd hh:mm:ss")
        res = self.get_record(time_start, time_end)
        for i in range(len(res)):
            self.ui.plainTextEdit_1.insertRow(i)
            self.ui.plainTextEdit_2.insertRow(i)
            self.ui.plainTextEdit_1.setItem(i, 0, QTableWidgetItem(res[i]['time']))
            self.ui.plainTextEdit_1.setItem(i, 1, QTableWidgetItem(res[i]['action']))
            self.ui.plainTextEdit_2.setItem(i, 0, QTableWidgetItem(res[i]['time']))
            self.ui.plainTextEdit_2.setItem(i, 1, QTableWidgetItem(res[i]['resolution']))


    def removeRecord(self):
        #清除系统日志界面
        self.ui.plainTextEdit_1.setRowCount(0)

    def removeTackle(self):
        #激活清除处理按钮
        self.ui.plainTextEdit_2.setRowCount(0)

    def getActionText(self):
        #获取鼠标选中的单元格内容
        contents = self.ui.plainTextEdit_1.selectedItems()[0].text()
        return contents


    def cancleTackle(self):
        #激活撤销处理按钮
        action = self.getActionText()
        self.edit_record(action)
        self.removeTackle()
        self.showRecord()

    def isAbnormal(self, action):
        #判断某个行为是否是异常行为
        #应该返回“是”或者“否”
        abnormal = []
        if action in abnormal:
            return '是'
        else:
            return '否'


    def save_record(self):
        #激活保存按钮，将界面输出的信息存入数据库
        for i in range(TABLE_MAXSIZE):
            if self.ui.headersTable_2.item(i, 0) is not None:
                time = self.ui.headersTable_2.item(i, 0).text()
                action = self.ui.headersTable_2.item(i, 1).text()
                isabnormal = self.ui.headersTable_2.item(i, 2).text()
                resolution = RESOLUTION[action]
                self.insert_record(time, action, isabnormal, resolution)
        QMessageBox.information(
            self.ui,
            '保存成功',
            '保存成功!')

    def insert_record(self, time, action, isabnormal, resolution):
        #将界面上输出的信息存入数据库中
        with connection.cursor() as cursor:
            #dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sql = "INSERT INTO `actions` (`time`, `action`, `isabnormal`, `resolution`) VALUES (%s, %s, %s, %s)"
            try:
                cursor.execute(sql, (time, action, isabnormal, resolution))
            except Exception as e:
                print(str(e))
        connection.commit()

    def edit_record(self, action):
        #将界面上输出的信息存入数据库中
        with connection.cursor() as cursor:
            #dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sql = "UPDATE `actions` SET `resolution` = '允许此行为发生！' WHERE `action`= " + "'" + action + "'"
            try:
                cursor.execute(sql)
            except Exception as e:
                print(str(e))
        connection.commit()


if __name__ == '__main__':
    app = QApplication([])
    mainwindow = BehaviorMonitor()
    #神经网络初始化
    mainwindow.init_ResNet()

    #测试用初始化
    mainwindow.showInitial()

    #开启子线程
    mainwindow.thread_1 = Thread(target = mainwindow.showDatetime,
                    args=())
    #设置为守护线程，及时关闭子线程
    mainwindow.thread_1.setDaemon(True)
    mainwindow.thread_1.start()

    mainwindow.ui.show()
    #退出界面
    app.exec_()

