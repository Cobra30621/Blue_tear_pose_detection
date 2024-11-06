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
def cal_vector2D_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try: 
        angle= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5)))) # 計算兩個向量之間的角度
    except: # 如果除以0，則角度為180度
        angle = 180
    return angle

# 根據傳入的 21 個節點座標，得到該手指的角度
def cal_hand_angles(hand): 
    angle_list = []
    for _, points in FINGER_ANGLE_POINTS.items():
        angle = cal_vector2D_angle(
        ((int(hand[points[0]].x)- int(hand[points[1]].x)),(int(hand[points[0]].y)-int(hand[points[1]].y))),
        ((int(hand[points[2]].x)- int(hand[points[3]].x)),(int(hand[points[2]].y)- int(hand[points[3]].y)))
        )
        angle_list.append(angle)
    return angle_list

def pose_detect(hand_angles):
    # results = [("彎曲", angle) if angle >= 50 else ("伸直", angle) for angle in hand_angles ]
    results = [False if angle >= 50 else True for angle in hand_angles ]
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



def hands_detect(image, tracked_hand=None):  # Add tracked_hand parameter
    width, height = image.shape[1], image.shape[0]
    pose = "noPose"
    hand_landmark_origin = None

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        if tracked_hand is None:  # No hand tracked yet
            hand_landmark = next(iter(results.multi_hand_landmarks)) # 獲取第一個檢測到的手的節點
            tracked_hand = copy.copy(hand_landmark) # Store the first detected hand
        else: # we are already tracking one hand
            min_dist = float('inf') # 初始化最小距離為無限大
            closest_hand = None # 初始化最接近的手為None
            for hand_landmark in results.multi_hand_landmarks: # 遍歷所有檢測到的手的節點
                dist = 0 # 初始化距離為0
                for i in range(21): # 遍歷所有節點
                    dist += (hand_landmark.landmark[i].x - tracked_hand.landmark[i].x)**2 + (hand_landmark.landmark[i].y - tracked_hand.landmark[i].y)**2
                dist = math.sqrt(dist)

                if dist < min_dist: # 如果當前距離小於最小距離
                    min_dist = dist # 更新最小距離
                    closest_hand = hand_landmark # 更新最接近的手

            if closest_hand is not None:
                hand_landmark = closest_hand # 更新手節點
                tracked_hand = copy.copy(hand_landmark) # 更新追蹤的手節點

        hand_landmark_origin = copy.copy(hand_landmark) # 複製手節點
        for i in hand_landmark.landmark:
            i.x = i.x * width
            i.y = i.y * height
        if hand_landmark.landmark:
            angle_list = cal_hand_angles(hand_landmark.landmark)
            pose = pose_detect(angle_list)

    elif tracked_hand:
        # Hand lost, reset tracked hand
        tracked_hand = None
        pose = "noPose"

    return pose, hand_landmark_origin, tracked_hand  # Return tracked_hand

def draw_hand_landmarks(image, hand_landmark):
    # 確認影像是可寫的
    image.setflags(write=1) # 將影象設置為可寫
    mp_drawing.draw_landmarks(image, hand_landmark, mp_hands.HAND_CONNECTIONS) # 在影象上繪製節點標註
    return image

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    tracked_hand = None  # Initialize tracked_hand outside the loop

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        pose, hand_landmark_origin, tracked_hand = hands_detect(image, tracked_hand) # pass and receive the tracked_hand

        if hand_landmark_origin:
            image = draw_hand_landmarks(image, hand_landmark_origin)
        
        print(pose) # or display pose in a nicer way within the image
        cv2.imshow('MediaPipe Holistic', image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()


