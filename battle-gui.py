import os
import pygame
from pygame import *
import cv2
import battle
import threading
import numpy as np
from sys import exit

sw, sh = 1920, 1020
background_image_filename = 'Naruto_background.jpg'
background_image_filename = os.path.join('res', background_image_filename)
mouse_image_filename = 'Naruto_mouse.jpg'
mouse_image_filename = os.path.join('res', mouse_image_filename)
#指定影象檔名稱

pygame.init()
#初始化pygame,為使用硬體做準備

screen = pygame.display.set_mode((sw, sh), 0, 32)
screen.fill((255, 255, 255))
#建立了一個視窗
pygame.display.set_caption("Hello, Naruto!")
#設定視窗標題

background = pygame.image.load(background_image_filename)#.convert()
background = pygame.transform.scale(background, (sw, sh))
target_h = battle.target_img_h
target_w = battle.target_img_w
mudra_h  = battle.player_showed_mudra_h
mudra_w  = battle.player_showed_mudra_w

target_img_h = 300
target_img_w = 400
player_showed_mudra_h = 450
player_showed_mudra_w = 600

A_target = battle.target_mudra_pygame['A']
B_target = battle.target_mudra_pygame['B']
A_mudra  = battle.player_showed_mudra['A']
B_mudra  = battle.player_showed_mudra['B']
#載入並轉換影象

game_on = False

def display_mudra():

    global screen
    global A_target
    global B_target
    global A_mudra
    global B_mudra
    global sh
    global sw
    global mudra_h
    global mudra_w

    A_target = battle.target_mudra_pygame['A']
    B_target = battle.target_mudra_pygame['B']
    A_mudra  = battle.player_showed_mudra['A']
    B_mudra  = battle.player_showed_mudra['B']

    if A_target is not None:
        screen.blit(A_target, (0,0))
    if B_target is not None:
        screen.blit(B_target, (sw-mudra_w, 0))

    if A_mudra is not None:
        screen.blit(A_mudra, (0, sh-mudra_h))
    if B_mudra is not None:
        screen.blit(B_mudra, (sw-mudra_w, sh-mudra_h))

while True:
#遊戲主迴圈

    for event in pygame.event.get():
        if event.type == QUIT:                                        # pygame.QUIT
            #接收到退出事件後退出程式
            exit()
        if event.type == KEYDOWN:                                     # pygame.KEYDOWN
            if event.key == K_q:
                exit()
        if event.type == KEYDOWN:                                     # pygame.KEYDOWN
            if event.key == K_KP_ENTER or event.key == K_RETURN:      # pygame.K_KP_ENTER / pygame.K_RETURN
                if game_on == False:
                    battle_t = threading.Thread(target=battle.battle, args=())
                    battle_t.start()
                    game_on = True

    screen.blit(background, (0,0))
    #將背景圖畫上去

    if game_on:
        display_mudra()


    pygame.display.update()
    #重新整理一下畫面
