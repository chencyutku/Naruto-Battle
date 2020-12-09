import os
import time as t
import pygame
from pygame import *
from color import *
import cv2
import battle
import threading
import numpy as np
from sys import exit

# 指定影象檔名稱
sw, sh = 1920, 1020
background_image_filename = 'Naruto_background.jpg'
background_image_filename = os.path.join('res', background_image_filename)
mouse_image_filename = 'Naruto_mouse.jpg'
mouse_image_filename = os.path.join('res', mouse_image_filename)

#初始化pygame,為使用硬體做準備
pygame.init()

# 建立了一個視窗
screen = pygame.display.set_mode((sw, sh), 0, 32)
# 填滿白底
screen.fill(WHITE)
# 設定視窗標題
pygame.display.set_caption("Hello, Naruto!")
# 設定主選單提示文字
ttf_path = os.path.join("res", "Microsoft JhengHei.ttf")
text_font = pygame.font.Font(ttf_path, 32)
start_game_text = text_font.render("press <Enter> to start game", True, BLACK, WHITE)
quit_text = text_font.render("press <q> to start game", True, BLACK, WHITE)


background = pygame.image.load(background_image_filename)#.convert()
background = pygame.transform.scale(background, (sw, sh))
target_h = battle.target_img_h
target_w = battle.target_img_w
mudra_h  = battle.player_showed_mudra_h
mudra_w  = battle.player_showed_mudra_w

A_target = battle.target_mudra_pygame['A']
B_target = battle.target_mudra_pygame['B']
A_mudra  = battle.player_showed_mudra['A']
B_mudra  = battle.player_showed_mudra['B']

head_w = 130
head_h = 120
A_head_filename = os.path.join('res', 'naruto_head.png')
A_head = pygame.image.load(A_head_filename).convert_alpha()
A_head = pygame.transform.scale(A_head, (head_w, head_h))
B_head_filename = os.path.join('res', 'sasuke_head.png')
B_head = pygame.image.load(B_head_filename).convert_alpha()
B_head = pygame.transform.scale(B_head, (head_w, head_h))
A_head_posx = target_w
A_head_posy = target_h
B_head_posx = sw-target_w-head_w
B_head_posy = target_h
A_HP_posx = target_w + 10
A_HP_posy = target_h - 20
B_HP_posx = sw - target_w - 100 - 10
B_HP_posy = target_h - 20

game_on = False
game_win = False

battle_t = threading.Thread(target=battle.battle, daemon=True, args=())


def draw_mudra():

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
        screen.blit(B_target, (sw-target_w, 0))

    if A_mudra is not None:
        screen.blit(A_mudra, (0, sh-mudra_h))
    if B_mudra is not None:
        screen.blit(B_mudra, (sw-mudra_w, sh-mudra_h))


def draw_head():

    global screen
    global sh, sw
    global A_head, B_head
    global A_head_posx, A_head_posy
    global B_head_posx, B_head_posy

    screen.blit(A_head, (A_head_posx, A_head_posy))
    screen.blit(B_head, (B_head_posx, B_head_posy))


def draw_HP():

    global screen
    global sh, sw
    global A_head, B_head
    global A_head_posx, A_head_posy
    global B_head_posx, B_head_posy
    global A_HP_posx, A_HP_posy
    global B_HP_posx, B_HP_posy

    for i in range(battle.HP_A):
        pygame.draw.rect(screen, RED, [A_HP_posx+i*105,A_HP_posy,100, 20],0)
    for i in range(battle.HP_B):
        pygame.draw.rect(screen, RED, [B_HP_posx-i*105,B_HP_posy,100, 20],0)



def draw_text(text, center: list):
    textRect = text.get_rect()
    textRect.center = center
    screen.blit(text, textRect)

def draw_win(winner):
    global ttf_path
    global game_on, game_win

    game_on = False
    game_win = False

    text_font = pygame.font.Font(ttf_path, 54)
    text = text_font.render("{} win the game !!!".format(winner), True, RED, WHITE)
    textRect = text.get_rect()
    textRect.center = (sw//2, sh//2)
    screen.blit(text, textRect)
    pygame.display.update()
    t.sleep(2)

while True:
# 遊戲視窗主迴圈

    for event in pygame.event.get():
        if event.type == QUIT:                                        # pygame.QUIT
            #接收到退出事件後退出程式
            battle.end_game = True
            # battle_t.join()
            pygame.quit()
            quit()
            # exit()
        if event.type == KEYDOWN:                                     # pygame.KEYDOWN
            if event.key == K_q:
                battle.end_game = True
                # battle_t.join()
                pygame.quit()
                quit()
                # exit()
        if event.type == KEYDOWN:                                     # pygame.KEYDOWN
            if event.key == K_KP_ENTER or event.key == K_RETURN:      # pygame.K_KP_ENTER / pygame.K_RETURN
                if game_on == False:
                    battle_t.start()
                    game_on = True

    if game_on == False:
        # 將背景圖畫上去
        screen.blit(background, (0,0))
        # 印出主選單
        draw_text(start_game_text, (sw//2, sh-sh//4))
        draw_text(quit_text, (sw//2, sh-sh//4+50))
    else:
        screen.fill(WHITE)
        draw_mudra()
        draw_head()
        draw_HP()
        if battle.end_game == True:
            battle_t.join()
            game_on = False
            battle.end_game = False
            draw_win(battle.winner)


    #重新整理一下畫面
    pygame.display.update()
