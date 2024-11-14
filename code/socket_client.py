#-*-coding:utf-8-*- 
from pose_detect import *
import threading
import cv2
import os
import time
import socket
import logging
import traceback
import numpy as np


def create_waiting_image(text="Waiting for camera to read", width=640, height=480, channels=3):
  """
  创建一个黑底白字的图像，显示指定的文字。

  参数:
      text (str): 要显示的文字，默认值为“等待攝影機讀取中”。
      width (int): 图像的宽度，默认值为640。
      height (int): 图像的高度，默认值为480。
      channels (int): 图像的通道数，默认值为3（彩色图像）。

  返回:
      image (numpy.ndarray): 创建的图像。
  """
  # 创建一个黑色背景的图像
  image = np.zeros((height, width, channels), dtype=np.uint8)

  # 设置文字内容、字体、大小和颜色
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 1
  font_color = (255, 255, 255)  # 白色
  font_thickness = 2

  # 计算文字的尺寸
  (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

  # 计算文字的位置，使其居中
  text_x = (width - text_width) // 2
  text_y = (height + text_height) // 2

  # 在图像上绘制文字
  cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

  # 将图像左右镜像反转
  mirrored_image = cv2.flip(image, 1)

  return mirrored_image

logging.basicConfig(level=logging.INFO, filename='log.txt', filemode='a+', # 設置日誌級別、文件名、文件模式
  format='[%(asctime)s %(levelname)-8s %(levelno)s] %(message)s', # 設置日誌格式
  datefmt='%Y%m%d %H:%M:%S' # 設置日期格式
  )

global_frame = create_waiting_image()
success = False
lock = threading.Lock()
pose_dict = ["noPose", "SunRight", "SunLeft", "CloseLight", "WindRight", "WindLeft", "Pray", "WindForward"]

HOST = '127.0.0.1'
PORT = 11000

is_connect = False
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # 創建一個TCP/IP套接字


def read_camera_setting(filename):
  with open(filename, 'r') as file:
    return int(file.read().strip())


def get_frame(): # 獲取攝像頭畫面
  global global_frame, success
  filename = 'camera_setting.txt'
  camera_index = read_camera_setting(filename)
  print(f"Camera index read from {filename}: {camera_index}")

  cap = cv2.VideoCapture(camera_index)

  while True:
    if not cap.isOpened():
      print("Camera is not opened. Reinitializing...")
      cap = cv2.VideoCapture(camera_index)
      if not cap.isOpened():
        success = False
        print("Failed to reinitialize the camera.")
        break

    success, frame = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    lock.acquire()
    global_frame = frame.copy()
    lock.release()

    # cv2.imshow("camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()

def test_connect(): # 測試連接
  global is_connect
  print("Testing Connect...")
  logging.info('Testing Connect...')
  while True:
    while not is_connect:
      try:
          s.settimeout(1)
          s.connect((HOST, PORT))
          is_connect = True
          print("Testing Connect Success")
          logging.info('Testing Connect Success')
      except socket.timeout as e:
        print("Testing Connect Timeout:", e)
        logging.warning("Testing Connect Timeout: " + str(e))
        time.sleep(2)
      except WindowsError as e:
        print(e)
        logging.error("Testing Connect WindowsError: " + str(e))
        if e.winerror == 10056:
          is_connect = True
      except Exception as e:
        print(e)
        logging.error("Testing Connect Failed: " + str(e))
    time.sleep(5)
        
def detect_frame(): # 檢測畫面
    global is_connect, s, global_frame, success
    tracked_hand = None  # Initialize tracked_hand outside the loop

    # 持續運行的主循環
    while True:
        try:
            # 初始化姿势为"noPose"
            poses = ["noPose"]
            
            # 获取当前帧的副本
            lock.acquire()
            try:
                image = global_frame.copy()
            except:
                # 如果获取帧失败，记录错误信息
                error_str = traceback.format_exc()
                print(error_str)
                logging.error("detect_frame Failed: " + str(error_str))
            lock.release()

            # 进行手势检测
            poses, hand_landmarks, tracked_hand = detect_hands(image, tracked_hand)

            # 如果攝像頭未成功捕獲畫面，將姿勢設置為"NotCaptureCamera"
            if not success:
                poses = ["NotCaptureCamera"]

            # 如果检测到手部标记点，在图像上绘制这些点
            for hand_landmark in hand_landmarks:
                image = draw_hand_landmarks(image, hand_landmark)

            # 绘制手部标记点的范围
            image = draw_hand_range(image)

            try:
                # 如果与服务器连接，发送检测到的姿势
                if is_connect:
                    s.sendall(bytes(','.join(poses), encoding='utf-8'))  # 将数组转换为以逗号分隔的字符串
                    data = s.recv(1024)
                    # 接收服务器的响应（当前未使用）
                    # print('Received', repr(data))
            except WindowsError as e:
                # 处理Windows特定的连接错误
                print(str(e))
                logging.error("socket connect Failed: " + str(e))
                is_connect = False
                s.close()
                # 重新创建socket对象
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            except Exception as e:
                # 处理其他可能的错误
                error_str = traceback.format_exc()
                print(error_str)
                logging.error("send message Failed: " + str(error_str))

            # 水平翻转图像（镜像效果）
            image = cv2.flip(image, 1)



            # 显示处理后的图像
            cv2.imshow('MediaPipe hands', image)

            # 检查是否按下'q'键来退出循环
            if cv2.waitKey(1) == ord('q'):
                break    # 按下 q 鍵停止


        except Exception as e:
            # 捕获并记录函数中可能出现的任何异常
            error_str = traceback.format_exc()
            print(error_str)
            logging.error("detect_frame Failed: " + str(error_str))

a = threading.Thread(target=get_frame)
a.daemon=True
a.start()
time.sleep(1)
b = threading.Thread(target=test_connect)
b.daemon = True
b.start()
time.sleep(1)
detect_frame()


# while True:
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.connect((HOST, PORT))
#         s.sendall(bytes(l[0], encoding='utf-8'))
#         data = s.recv(1024)

#     print('Received', repr(data))
#     time.sleep(0.5)