import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import cvzone
#Define class
class Ques_tion():
    def __init__(self,data):
        self.question = data[0]
        self.data1 = data[1]
        self.data2 = data[2]
        self.data3 = data[3]
        self.data4 = data[4]
        self.ans = int(data[5])

        self.UserAns = None

    def click(self, cursor, bboxlist):
        for x, bbox in enumerate(bboxlist):
            x1, y1, x2, y2, = bbox
            if x1<cursor[0]<x2 and y1<cursor[1]<y2:
                self.UserAns = x+1
                cv2.rectangle(image, (x1,y1), (x2,y2), (255,255,255), cv2.FILLED)
WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)
ques=0
ques_list=[]
score=0
question_df=pd.read_csv(r"Quiz.csv",encoding="utf-8",sep=";")
ques_tot=question_df.shape[0]
for j in range (0,ques_tot):
    ques_list.append(Ques_tion(question_df.loc[j,:]))
# to draw the landmarks
mp_draw = mp.solutions.drawing_utils

# Set the drawing specs.
# the defaults also look fine.
draw_specs = mp_draw.DrawingSpec(thickness=2, circle_radius=2)

mp_pose = mp.solutions.hands
pose = mp_pose.Hands(static_image_mode=False,
               max_num_hands=2,
               min_detection_confidence=0.7,
               min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
video_width = 720
video_height = 480
cap.set(3, video_width) # 3 is the id for width
cap.set(4, video_height) # 4 is the id for height


start_time=100
while True:
    # Get one frame.
    # # 'image' is the image frame that was read.
    success, image = cap.read()
    image = cv2.flip(image, 1)
    # Convert the image from BGR to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # process the image and store the results in an object
    results = pose.process(image_rgb)
    current_time = time.time()
    diff=current_time-start_time
    if ques < ques_tot:
        cv2.rectangle(image, (20, 15), (120, 60), (137,165,23), cv2.FILLED)
        cv2.putText(image, str(start_time), (40, 50), cv2.FONT_HERSHEY_PLAIN,2,WHITE_COLOR, 3)
        image, bboxq = cvzone.putTextRect(image, ques_list[ques].question, [80,120], 1.3, 2, offset = 25, border = 2,colorR = (137,165,23),colorB= (255, 255, 255))
        image, bbox1 = cvzone.putTextRect(image, str(ques_list[ques].data1), [120, 250],1.5, 2, offset=30, border=0,colorR = (137,165,23),colorB= (255, 255, 255))
        image, bbox2 = cvzone.putTextRect(image, str(ques_list[ques].data2), [450, 250], 1.5,2, offset=30, border=0,colorR = (137,165,23),colorB= (255, 255, 255))
        image, bbox3 = cvzone.putTextRect(image, str(ques_list[ques].data3), [120, 400], 1.5, 2, offset=30, border=0,colorR = (137,165,23),colorB= (255, 255, 255))
        image, bbox4 = cvzone.putTextRect(image, str(ques_list[ques].data4), [450, 400], 1.5, 2, offset=30, border=0,colorR = (137,165,23),colorB= (255, 255, 255))
        liste=[bbox1, bbox2, bbox3, bbox4]
        if results.multi_hand_landmarks:
            for hand_number,hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(image, hand_landmarks, mp_pose.HAND_CONNECTIONS, mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))
                coords = tuple(np.multiply(np.array((hand_landmarks.landmark[mp_pose.HandLandmark.INDEX_FINGER_TIP].x, hand_landmarks.landmark[mp_pose.HandLandmark.INDEX_FINGER_TIP].y)),[640,480]).astype(int))
            start_time=start_time-1
            if start_time==0 and ques_list[ques].UserAns==None:
                print("You leave the game")
                break
            elif start_time!=0 and start_time<=80:
                ques_list[ques].click(coords, liste)
                if ques_list[ques].UserAns!=None:
                    r1,j1,r2,j2=liste[ques_list[ques].ans-1]
                    if ques_list[ques].UserAns==ques_list[ques].ans:
                        score=score+10
                        cv2.rectangle(image, (r1,j1), (r2,j2), (0,255,0), cv2.FILLED)
                    else:
                        cv2.rectangle(image, (r1,j1), (r2,j2), (0,255,0), cv2.FILLED)
                    time.sleep(0.5)
                    ques=ques+1
                    start_time=100

    else:
        image, _ = cvzone.putTextRect(image, "Quiz is completed", [180, 150], 2, 2, offset=30, border=3,colorR = (137,165,23),colorB= (255, 255, 255))
        image, _ = cvzone.putTextRect(image, f'Your score: {score}', [200, 300], 2, 2,offset=30, border=3,colorR = (137,165,23),colorB= (255, 255, 255))

    cv2.imshow('Video', image)



	# Press q on the keyboard to close the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
