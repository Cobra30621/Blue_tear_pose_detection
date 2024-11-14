from AngleDef import *
import mediapipe as mp
import math
import cv2
import copy

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 根據兩點的座標，計算角度
def calculate_vector_angle(vector1, vector2):
    """
    計算兩個向量之間的角度。
    
    參數:
    vector1: 第一個向量的座標 (x, y)
    vector2: 第二個向量的座標 (x, y)
    
    返回:
    兩個向量之間的角度（度數）。
    """
    vector1_x, vector1_y = vector1
    vector2_x, vector2_y = vector2
    try: 
        angle = math.degrees(math.acos((vector1_x * vector2_x + vector1_y * vector2_y) /
                                        (math.sqrt(vector1_x**2 + vector1_y**2) * 
                                         math.sqrt(vector2_x**2 + vector2_y**2))))
    except ZeroDivisionError:  # 如果除以0，則角度為180度
        angle = 180
    return angle

# 根據傳入的 21 個節點座標，得到該手指的角度
def calculate_hand_angles(hand_landmarks): 
    """
    根據手的節點座標計算手指的角度。
    
    參數:
    hand_landmarks: 手的 21 個節點的座標。
    
    返回:
    包含每個手指角度的列表。
    """
    angles = []
    for _, points in FINGER_ANGLE_POINTS.items():
        angle = calculate_vector_angle(
            (int(hand_landmarks[points[0]].x) - int(hand_landmarks[points[1]].x),
             int(hand_landmarks[points[0]].y) - int(hand_landmarks[points[1]].y)),
            (int(hand_landmarks[points[2]].x) - int(hand_landmarks[points[3]].x),
             int(hand_landmarks[points[2]].y) - int(hand_landmarks[points[3]].y))
        )
        angles.append(angle)
    return angles

def detect_pose(hand_angles):
    """
    根據手指的角度檢測手勢。
    
    參數:
    hand_angles: 手指的角度列表。
    
    返回:
    檢測到的手勢名稱。
    """
    results = [angle < 50 for angle in hand_angles]
    if results == [True, True, False, False, False]:
        return "WindForward"
    elif results == [True, False, False, False, False]:
        return "Pray"
    elif results == [False, True, True, False, False]:
        return "SunRight"
    elif results == [False, False, False, False, False]:
        return "CloseLight"
    else:
        return "noPose"

# 新增：讀取hand_range.txt檔案，並將長方形座標存到hand_range中
def load_hand_range(file_path):
    """
    從指定的檔案中讀取手的範圍座標。
    
    參數:
    file_path: 檔案路徑。
    
    返回:
    包含手範圍的字典，包含 'x' 和 'y' 座標。
    """
    hand_range = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('x:'):
                hand_range['x'] = list(map(float, line.strip().split(':')[1].split()))
            elif line.startswith('y:'):
                hand_range['y'] = list(map(float, line.strip().split(':')[1].split()))
    return hand_range

hand_range = load_hand_range('hand_range.txt')  # 新增：讀取hand_range.txt

# 新增：在影像上畫出hand_range的範圍
def draw_hand_range(image):
    """
    在影像上繪製手的範圍矩形。
    
    參數:
    image: 要繪製的影像。
    
    返回:
    繪製了手範圍的影像。
    """
    image.flags.writeable = True  # 確保影像是可寫的
    cv2.rectangle(image,
                  (int(hand_range['x'][0] * image.shape[1]), int(hand_range['y'][0] * image.shape[0])),
                  (int(hand_range['x'][1] * image.shape[1]), int(hand_range['y'][1] * image.shape[0])),
                  (0, 255, 0), 2)  # 繪製綠色矩形
    return image

# 篩選在 hand_range 中的手座標
def filter_hands_in_range(multi_hand_landmarks):
    """
    篩選在指定範圍內的手座標。
    
    參數:
    multi_hand_landmarks: 檢測到的多個手的節點。
    
    返回:
    在範圍內的手的列表。
    """
    if not multi_hand_landmarks:
        return []

    filtered_hands = []
    for hand_landmark in multi_hand_landmarks:
        x_coords = [lm.x for lm in hand_landmark.landmark]
        y_coords = [lm.y for lm in hand_landmark.landmark]
        if (hand_range['x'][0] <= min(x_coords) <= hand_range['x'][1] and
            hand_range['y'][0] <= min(y_coords) <= hand_range['y'][1]):
            filtered_hands.append(hand_landmark)
    return filtered_hands

def detect_hands(image, tracked_hand=None):  # Add tracked_hand parameter
    """
    檢測影像中的手並返回手勢和手的節點。
    
    參數:
    image: 要檢測的影像。
    tracked_hand: 目前追蹤的手的節點。
    
    返回:
    檢測到的手勢和手的節點。
    """
    width, height = image.shape[1], image.shape[0]
    poses = []  # 新增：用於儲存所有手勢的列表
    hand_landmarks = []  # 新增：用於儲存所有手的原始節點

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    multi_hand_landmarks = filter_hands_in_range(results.multi_hand_landmarks)
    for hand_landmark in multi_hand_landmarks:
        hand_landmark_origin = copy.copy(hand_landmark)  # 複製手節點
        for landmark in hand_landmark.landmark:
            landmark.x *= width
            landmark.y *= height
        if hand_landmark.landmark:
            angle_list = calculate_hand_angles(hand_landmark.landmark)
            pose = detect_pose(angle_list)
            poses.append(pose)  # 新增：將檢測到的手勢加入列表
            hand_landmarks.append(hand_landmark_origin)  # 新增：將手的原始骨架加入列表

    hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks is not None else 0
    print(f"鏡頭中手數量 :{hand_count}, "
          f"在條件框的手數量: {len(multi_hand_landmarks)}, "
          f"動作: {poses}")

    return poses, hand_landmarks, tracked_hand  # 修改：回傳手勢列表和手的原始節點列表

def find_closest_hand(multi_hand_landmarks, tracked_hand):
    """
    找到與追蹤手最近的手。
    
    參數:
    multi_hand_landmarks: 檢測到的多個手的節點。
    tracked_hand: 目前追蹤的手的節點。
    
    返回:
    最近的手的節點。
    """
    min_dist = float('inf')  # 初始化最小距離為無限大
    closest_hand = None  # 初始化最接近的手為None
    for hand_landmark in multi_hand_landmarks:  # 遍歷所有檢測到的手的節點
        dist = sum((hand_landmark.landmark[i].x - tracked_hand.landmark[i].x) ** 2 +
                   (hand_landmark.landmark[i].y - tracked_hand.landmark[i].y) ** 2 for i in range(21))
        dist = math.sqrt(dist)

        if dist < min_dist:  # 如果當前距離小於最小距離
            min_dist = dist  # 更新最小距離
            closest_hand = hand_landmark  # 更新最接近的手
    return closest_hand

def draw_hand_landmarks(image, hand_landmark):
    """
    在影像上繪製手的節點標註。
    
    參數:
    image: 要繪製的影像。
    hand_landmark: 要繪製的手的節點。
    
    返回:
    繪製了手節點的影像。
    """
    # 確認影像是可寫的
    image.setflags(write=1)  # 將影象設置為可寫
    mp_drawing.draw_landmarks(image, hand_landmark, mp_hands.HAND_CONNECTIONS)  # 在影象上繪製節點標註
    return image

