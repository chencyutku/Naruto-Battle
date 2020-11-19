#region     [Import packages]

# import systemwise package
import os
import shutil
import queue
from queue import PriorityQueue as pq
import threading

# import basic tools
import random
import time

# import third-party package
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing import image
import pandas as pd
import numpy as np
import pygame


#endregion  [Import packages]


#region     [Global variables]

# 設定
# two_player = False
two_player = True
play_round = 2
HP_A = play_round
HP_B = play_round
ATK_A = False
ATK_B = False
# Only admit mudra if the probability greater than 70%
qualification = 0.6

# IMG should showed in GUI window
target_img_h = 300
target_img_w = 400
player_showed_mudra_h = 450
player_showed_mudra_w = 600
target_mudra_pygame = dict.fromkeys(['A','B'])
player_showed_mudra = dict.fromkeys(['A','B'])

cwd = os.getcwd()
workspace_dir = os.path.join(cwd, 'workspace')


label_list = [
    "bird",
    "boar",
    "dog",
    "dragon",
    "horse",
    "monkey",
    "ox",
    "rabbit",
    "rat",
    "sheep",
    "snake",
    "tiger"
]

purpose_dict = {
    "train":"",
    "validation":"",
    "test":""
}


#region     [神經網路輸入的圖片尺寸]
img_x = 150
img_y = 150
#endregion  [神經網路輸入的圖片尺寸]


#region     [遊戲結束的Flag]
end_game = False
#endregion  [遊戲結束的Flag]

# 忍術結印表-字典
# generate by
# ````Python
# for i in range(20):
#     random.sample(label_list,5)
# ````
mudras_num_in_ninjutsu = 3
Ninjutsu_mudras_dict = {
    "A":['rabbit', 'tiger' , 'dragon'   ],
    "B":['rabbit', 'tiger' , 'horse'    ],
    "C":['dog'   , 'sheep' , 'rabbit'   ],
    "D":['rat'   , 'ox'    , 'horse'    ],
    "E":['monkey', 'ox'    , 'boar'     ],
    "F":['snake' , 'sheep' , 'dog'      ],
    "G":['rat'   , 'dog'   , 'bird'     ],
    "H":['boar'  , 'horse' , 'dog'      ],
    "I":['horse' , 'tiger' , 'rabbit'   ],
    "J":['rat'   , 'dragon', 'dog'      ],
    "K":['boar'  , 'ox'    , 'rabbit'   ],
    "L":['boar'  , 'tiger' , 'monkey'   ],
    "M":['bird'  , 'ox'    , 'rabbit'   ],
    "N":['sheep' , 'rabbit', 'monkey'   ],
    "O":['sheep' , 'dragon', 'monkey'   ],
    "P":['monkey', 'snake' , 'dog'      ],
    "Q":['ox'    , 'bird'  , 'monkey'   ],
    "R":['bird'  , 'rabbit', 'dog'      ],
    "S":['bird'  , 'dragon', 'ox'       ],
    "T":['ox'    , 'bird'  , 'monkey'   ]
}


# 忍術題目串列
Ninjutsu_List  = list()


# 設定結印圖片的路徑
image_path = 'images'
Mudra_dir_path = os.path.join(image_path, 'mudras')


# 神經網路模型存放路徑
model_dir = os.path.join(workspace_dir, 'train-logs')
best_model    = 'ninja-mudras-model'
best_model    = os.path.join(model_dir, best_model)

# 設定各個狀態的旗標
# global end_game = False


#endregion  [Global]


#region     [Declare Object]


# 接收攝影機串流影像，採用多執行緒的方式，降低緩衝區堆疊圖幀的問題。
class Camera:
    def __init__(self, webcam):
        global player_showed_mudra_h
        global player_showed_mudra_w
        self.webcam = webcam
        self.Frame = np.zeros((player_showed_mudra_h,player_showed_mudra_w,3), np.uint8)
        self.status = False
        self.isstop = False
		
	# 攝影機連接。
        self.capture = cv2.VideoCapture(webcam, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,  160)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    def start(self):
	# 把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        print('ipcam{} started!'.format(self.webcam))
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
	# 記得要設計停止無限迴圈的開關。
        self.isstop = True
        print('ipcam{} stopped!'.format(self.webcam))
   
    def getframe(self):
	# 當有需要影像時，再回傳最新的影像。
        return self.Frame
        
    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()
            cv2.waitKey(30)
        
        self.capture.release()


#endregion  [Declare Object]


#region     [Declare Function]


def create_model():

    inputs = keras.Input(shape=(img_x, img_y, 3))
    x = layers.Conv2D( 64, (5,5), activation='relu')(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, (5,5), activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, (5,5), activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D( 64, (5,5), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(len(label_list),       activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(label_list), activation='softmax')(x) # 輸出神經元個數採用我們的label數量

    model = keras.Model(inputs, outputs, name='model')
    # model.summary()  # 查看模型摘要

    # %% [markdown]
    # 
    # ### 建立model儲存資料夾
    # 

    # %%
    model_dir = os.path.join(workspace_dir, 'train-logs')

    # model_dir 不存在時，建立新資料夾
    if not (os.path.isdir(model_dir)): 
        os.mkdir(model_dir)

    # %% [markdown]
    # 
    # ### 建立回調函數(Callback function)
    # 

    # %%
    # 將訓練紀錄儲存為TensorBoard的紀錄檔
    log_dir = os.path.join(model_dir, 'model')
    model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)

    # 儲存最好的網路模型權重
    best_model_h5 = 'ninja-mudras-model-best.h5'
    best_model_h5 = os.path.join(model_dir, best_model_h5)
    model_mckp = keras.callbacks.ModelCheckpoint(best_model_h5,
                                                monitor='val_categorical_accuracy',
                                                save_best_only=True,
                                                mode='max')


    # 問：改成多元分類是否要更改損失函數？
    # 答：參考他人專案改成 'categorical_crossentropy'
    model.compile(optimizer=optimizers.Adam(),
                loss=losses.CategoricalCrossentropy(),
                metrics=[metrics.CategoricalAccuracy()])

    model.load_weights(best_model_h5)
    
    return model


def wait_queue_if_empty(q:queue):
    global end_game
    wait = q.empty()
    while wait:
        if end_game == True:
            break
        print("wait if empty.")
        wait = q.empty()
        time.sleep(0.05)


def wait_queue_if_not_empty(q:queue):
    global end_game
    wait = q.empty()
    while not wait:
        if end_game == True:
            break
        print("wait if not empty.")
        wait = q.empty()
        time.sleep(0.05)

def Img2Pygame(frame):
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    frame = np.rot90(frame)
    frame = pygame.surfarray.make_surface(frame)
    return frame

def show_victory(player:str):

    img = np.zeros((1920, 1020, 3), np.uint8)
    img.fill(90)

    text = "player " + player + " win the game !!!"

    cv2.putText(img, text, (300, 300), cv2.FONT_HERSHEY_SIMPLEX,  1, (0, 255, 255), 1, cv2.LINE_AA)
    # cv2.namedWindow('win')
    # cv2.moveWindow('win', 660, 240)
    # cv2.startWindowThread()
    # cv2.imshow('win', img)

    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()


def detect_mudra(webcam:int, player:str, target_queue:queue.Queue, answer_queue:queue.Queue):
    # Continuous detect mudra from webcam

    # Only admit mudra if the probability greater than 70%
    global qualification

    global end_game
    global player_showed_mudra
    global player_showed_mudra_h
    global player_showed_mudra_w
    global model

    camera = Camera(webcam)
    camera.start()
    time.sleep(1)

    while end_game == False:

        

        # print("detect")

        # 從WebCam讀取一張圖片
        frame = camera.getframe()
        frame_show = cv2.resize(frame, (player_showed_mudra_w, player_showed_mudra_h))
        player_showed_mudra[player] = Img2Pygame(frame_show)
        # cv2.namedWindow(player)        # Create a named window
        # if player == 'A':
        #     cv2.moveWindow(player, 40,540)  # Move it to (40,30)
        # else:
        #     cv2.moveWindow(player, 1480,540)
        # cv2.startWindowThread()
        # cv2.imshow(player, frame_show)
        
        cv2.waitKey(30)

        print("detect")

        
        wait_queue_if_empty(target_queue)
        if end_game == True:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_x, img_y), interpolation=cv2.INTER_NEAREST)
        frame = np.array(frame) / 255
        frame = image.img_to_array(frame)
        frame = np.expand_dims(frame, axis = 0)

        predict_output = model.predict(frame)               # 輸出測試結果
        max_probablility_index = predict_output.argmax()    # 取得最高機率的index
        mudra_str = label_list[max_probablility_index]      # 根據index得到對應的mudra label
        max_probablility = np.max(predict_output)           # 取得最高機率數值

        if max_probablility > qualification:                # 達到標準才視為結印
            print("Player ", player, ": ", mudra_str)
            print(end_game)
            answer_queue.put(mudra_str)
            continue

        answer_queue.put('none')
    
    camera.stop()
    print("end detect_mudra {}".format(player))
    # cv2.destroyWindow('frame')


def judge_mudra(target_queue:queue.Queue, answer_queue:queue.Queue):

    end_ninjutsu = False
    global end_game
    global mudras_num_in_ninjutsu

    mudra_order = 0
    while (end_ninjutsu == False) and (end_game == False):

        print("judge")

        wait_queue_if_empty(target_queue)
        if end_game == True:
            break
        target_mudra = target_queue.queue[0]
        wait_queue_if_not_empty(answer_queue)
        if end_game == True:
            break
        answer_mudra = answer_queue.get()

        if target_mudra == answer_mudra:
            mudra_order += 1
            wait_queue_if_empty(target_queue)
            if end_game == True:
                break
            target_queue.get()
        
        if mudra_order >= mudras_num_in_ninjutsu:
            end_ninjutsu = True
    print("end judge.")


def play(webcam:int, player:str, target_queue:queue.Queue, answer_queue:queue.Queue):

    detect_mudra_thread = threading.Thread(target=detect_mudra, daemon=True, args=(webcam, player, target_queue, answer_queue))
    detect_mudra_thread.start()

    global end_game
    global play_round
    global HP_A
    global HP_B
    global ATK_A
    global ATK_B

    ninjutsu_complete_int = 0
    while end_game == False:

        judge_mudra(target_queue, answer_queue)
        ninjutsu_complete_int += 1
        if player == 'A':
            ATK_B = True
            HP_B -= 1
        if player == 'B':
            ATK_A = True
            HP_A -= 1

        if ninjutsu_complete_int >= play_round:
            show_victory(player)
            end_game = True
    print("squeeze  {}.".format())
    detect_mudra_thread.join()
    print("detect_mudra {}".format(player))
    print("end play {}".format(player))


def show_target_mudra(player:str, target_queue:queue.Queue):
    # 根據忍術顯示第(1~5)張
        # 傳輸WebCam影像給神經網路
        # 神經網路回傳判斷機率
        # 機率在合格值以上時PASS，迴圈繼續下一張結印圖片

    global Ninjutsu_mudras_dict
    global Mudra_dir_path
    global Ninjutsu_List
    global target_img_h
    global target_img_w
    global target_mudra_pygame
    global end_game

    if player == 'A':
        pos_x, pos_y = 0,0
    else:
        pos_x, pos_y = 1520,0
    for Ninjutsu_str in Ninjutsu_List:

        for mudra_str in Ninjutsu_mudras_dict[Ninjutsu_str]:

            Mudra_img_path = os.path.join(Mudra_dir_path, (mudra_str + '.jpg'))
            # print("show [", mudra_str, "] picture:")
            print(Mudra_img_path)
            # if os.path.isfile(Mudra_img_path):
            Mudra_img = cv2.imread(Mudra_img_path)
            # cv2.startWindowThread()

            print("squeeze{} 1.".format(player))
            
            Mudra_img = cv2.resize(Mudra_img, (target_img_w, target_img_h))
            # cv2.namedWindow(Ninjutsu_str)        # Create a named window
            # cv2.moveWindow(Ninjutsu_str, pos_x, pos_y)
            # cv2.startWindowThread()
            # cv2.imshow(Ninjutsu_str, Mudra_img)
            cv2.waitKey(500)

            print("squeeze{} 2.".format(player))

            # 將題目交給 pygame 顯示
            target_mudra_pygame[player] = Img2Pygame(Mudra_img)

            print("squeeze{} 3.".format(player))

            wait_queue_if_not_empty(target_queue)
            print("eng_game flag == ", end_game)
            if end_game == True:
                print("leave mudra {}.".format(player))
                break
            target_queue.put(mudra_str)

        print("eng_game flag == ", end_game)
        if end_game == True:
            print("leave ninjutsu {}.".format(player))
            break
        # cv2.destroyWindow(Ninjutsu_str)
    print("end show_target_mudra {}".format(player))


#endregion  [Declare Function]

model = tf.keras.models.load_model(best_model)

def battle():

    global HP_A
    global HP_B
    global play_round
    global mudras_num_in_ninjutsu

    HP_A = play_round
    HP_B = play_round

    target_queue_A = queue.Queue(maxsize=1)
    answer_queue_A = queue.Queue(maxsize=mudras_num_in_ninjutsu)
    if two_player:
        target_queue_B = queue.Queue(maxsize=1)
        answer_queue_B = queue.Queue(maxsize=mudras_num_in_ninjutsu)


    playerA_t = threading.Thread(target=play, daemon=True, args=(1,'A', target_queue_A, answer_queue_A))
    playerA_t.start()
    if two_player:
        playerB_t = threading.Thread(target=play, daemon=True, args=(2,'B', target_queue_B, answer_queue_B))
        playerB_t.start()

    # while global end_game == False:

    # 隨機產生play_round個指定忍術
    for i in range(play_round):
        Ninjutsu_str = random.choice(str().join(Ninjutsu_mudras_dict.keys()))
        Ninjutsu_List.extend(Ninjutsu_str)

    # print("忍術題目：", Ninjutsu_List)

    show_target_mudra_A_t = threading.Thread(target=show_target_mudra, daemon=True, args=('A', target_queue_A))
    show_target_mudra_A_t.start()
    if two_player:
        show_target_mudra_B_t = threading.Thread(target=show_target_mudra, daemon=True, args=('B', target_queue_B))
        show_target_mudra_B_t.start()


    show_target_mudra_A_t.join()
    print("show_target_mudra A join.")
    if two_player:
        show_target_mudra_B_t.join()
        print("show_target_mudra B join.")
        playerB_t.join()
        print("playerB_t join.")
    playerA_t.join()
    print("playerA_t join.")

    print("end battle.")

    # cv2.destroyAllWindows()


if __name__ == '__main__':
    battle()
