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

logging.basicConfig(level=logging.INFO, filename='log.txt', filemode='a+',
  format='[%(asctime)s %(levelname)-8s %(levelno)s] %(message)s',
  datefmt='%Y%m%d %H:%M:%S',
  encoding='utf-8'
  )

global_frame = create_waiting_image()
success = False
lock = threading.Lock()
pose_dict = ["noPose", "SunRight", "SunLeft", "CloseLight", "WindRight", "WindLeft", "Pray", "WindForward"]

HOST = '127.0.0.1'
PORT = 11000

is_connect = False
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def read_camera_setting(filename):
  with open(filename, 'r') as file:
    return int(file.read().strip())


def get_frame():
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

def test_connect():
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
        
def detect_frame():
    global is_connect, s, global_frame, success

    # For webcam input:
    while True:
      try:
        pose = "noPose"
        lock.acquire()
        try:
          image = global_frame.copy()
        except:
          error_str = traceback.format_exc()
          print(error_str)
          logging.error("detect_frame Failed: " + str(error_str))
        lock.release()

        pose, hand_landmark = hands_detect(image)
        if not success:
          pose = "NotCaptureCamera"

        if hand_landmark:
          image = draw_hand_landmarks(image, hand_landmark)

        try:
          if is_connect :
            s.sendall(bytes(pose, encoding='utf-8'), )
            data = s.recv(1024)
            # print('Received', repr(data))
        except WindowsError as e:
          print(str(e))
          logging.error("socket connect Failed: " + str(e))
          is_connect = False
          s.close()
          s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except Exception as e:
          error_str = traceback.format_exc()
          print(error_str)
          logging.error("send message Failed: " + str(error_str))

        image = cv2.flip(image, 1)
        cv2.imshow('MediaPipe hands', image)

        if cv2.waitKey(1) == ord('q'):
            break    # 按下 q 鍵停止
        print(pose)

      except Exception as e:
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