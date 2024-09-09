
# 藍眼淚手勢偵測

我們與曹松清畫家合作設計出以馬祖藍眼淚為主題的互動式畫展，畫作內可以以手勢來控制畫作的光影變化隨時欣賞藍眼淚的奇異美景


## Pre-Requisites

Install package for the project

```bash
  pip install requirements.txt
```


## Gestures
    1. 控制日落: 請將手呈現 ”剪刀” 的姿勢，並與鏡頭平行。(等待太陽下山)
    2. 開啟燈塔: 請將手呈現 ”布” 的姿勢，並與鏡頭平行。
    3. 關閉燈塔: 手心面對畫作，握拳緩緩關閉燈塔。
    4. 祈禱: 請將手呈現 ”讚” 的姿勢，手背朝前，並與鏡頭平行。。
    5. 放大蘭眼淚: 將手的拇指與食指緩緩張開，並與鏡頭平行。
    6. 縮小藍眼淚: 將手的拇指與食指緩緩靠近，並與鏡頭平行 。

## Setting
- camera_setting.txt: 裡面放攝影機編號(預設為 0)

## Demo
[![IMAGE ALT TEXT](http://img.youtube.com/vi/FBhia4bdPUk/0.jpg)](https://www.youtube.com/watch?v=FBhia4bdPUk)

## Export
- 主程式: socket_client.py
- 在終端機執行:
```
cd code
pyinstaller -F --add-data "D:\Unity\Blue_tear_pose_detection\venv\Lib\site-packages\mediapipe;mediapipe/" socket_client.py
```


## Unity Project Link
Unity打包於以下連結↓\
https://github.com/MasterSeafood-SeafoodMaster/unity_BlueTears
