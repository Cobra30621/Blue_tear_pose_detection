import cv2

def test_camera_indexes():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

available_cameras = test_camera_indexes()
print(f"可用的摄像头索引: {available_cameras}")

if available_cameras:
    print("请在camera_setting.txt文件中使用其中一个索引")
else:
    print("未检测到可用的摄像头")