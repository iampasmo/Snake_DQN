
#%

RED = 225, 50, 10       
GREEN = 30, 200, 100     
BLUE = 0, 0, 255      
PURPLE = 127, 0, 127  
BLACK = 0, 0, 0       
GRAY = 127, 127, 127   
WHITE = 255, 255, 255 

color_snake = GREEN * 1
color_apple = RED * 1
#color_background = BLACK * 1

color_background = (0,0,0)
color_bg_info = (100,50, 150)


# 번쩍번쩍 클래스
class Light:
    def __init__(self, amp = 255, rate = 1):        
        self.on(amp, rate)
        self.switch = False
        
    def _fire(self, amp = 255, rate = 1):        
        for i in np.arange(amp, 0, -rate):
            yield i
        while True:
            yield 0
            
    def on(self, amp = 255, rate = 1, switch = False):        
        self.light_source = self._fire(amp= amp, rate = rate)
        self.switch = switch
            
    def light(self):
        if self.switch == True:
            return self.light_source.__next__()
        else :
            return 0

def apple_color_random():
    color = (random.randint(200,250),  random.randint(60,120), random.randint(0,100)    )
    return color

    

L1 = Light(amp = 100)
L2 = Light(amp = 100)
L3 = Light(amp = 255)


import pygame             
import time

from pygame import gfxdraw

from datetime import datetime
from datetime import timedelta
import numpy as np
import copy

import random
            
# 사이즈
BLOCK_SIZE = 20
WIDTH = 25
HEIGHT = 25

# 그리기용 변수 
SCREEN_WIDTH = BLOCK_SIZE * WIDTH           # 게임 화면의 너비
SCREEN_HEIGHT = BLOCK_SIZE * HEIGHT         # 게임 화면의 높이

SCREEN_INFO_WIDTH = 150
info_joystick_height = 60

# 뱀 이동 횟수 제한 - 사과 먹을때 증가량
snake_move_increase = (WIDTH + HEIGHT)*2


# 그리기 관련 기본 함수들
def draw_background(screen):
    """게임의 배경을 그린다."""
    background = pygame.Rect((0, 0), (SCREEN_WIDTH, SCREEN_HEIGHT))        
    pygame.draw.rect(screen, color_background, background)      # 그림을 그릴땐 screen에 그려야 한다.
    
def draw_bg_info(screen):
    background = pygame.Rect((SCREEN_WIDTH, 0), (SCREEN_INFO_WIDTH, SCREEN_HEIGHT))        
    pygame.draw.rect(screen, color_bg_info, background)      

def draw_block_game_coor(screen, color, position ):
    """position 위치에 color 색깔의 블록을 그린다 - 게임상 좌표 이용"""
    block = pygame.Rect((position[0] * BLOCK_SIZE, position[1] * BLOCK_SIZE),
                        (BLOCK_SIZE-1, BLOCK_SIZE-1))
    pygame.draw.rect(screen, color, block)
    
def draw_block(screen, color, position, size = BLOCK_SIZE ):
    """position 위치에 color 색깔의 블록을 그린다 - 원래 좌표 이용"""
    block = pygame.Rect((position[0], position[1]), (size, size) )
    pygame.draw.rect(screen, color, block)
    
def draw_bar(screen, color, position, size_wh  ):
    """position 위치에 color 색깔의 막대기를 그린다."""
    if size_wh[1] >=0 :
        block = pygame.Rect( (position[0], position[1]-size_wh[1] ), size_wh )
        pygame.draw.rect(screen, color, block)
    else :
        block = pygame.Rect( (position[0], position[1] ), [size_wh[0], abs(size_wh[1])] )
        pygame.draw.rect(screen, color, block)
 

#% 게임 환경 설정
Time_delta = timedelta(seconds = 0.05)   

class Snake:
    """뱀 클래스"""
    color = color_snake  # 뱀 색깔
    
    # 키 입력 들어왔을 때 속도 반환하기 위한 딕셔너리
    key_direction_human = dict([(pygame.K_UP,    [ 0,-1]),
                          (pygame.K_DOWN,  [ 0, 1]),
                          (pygame.K_LEFT,  [-1, 0]),
                          (pygame.K_RIGHT, [ 1, 0]) 
                         ])
    
    key_direction_human2 = dict([(pygame.K_UP,     0),                          
                                 (pygame.K_RIGHT,  1),
                                 (pygame.K_LEFT,  -1)
                                 ])

    key_direction_ai = dict([(0, [ 0,-1]),
                             (1, [ 0, 1]),
                             (2, [-1, 0]),
                             (3, [ 1, 0]),
                             ])
    
    key_direction_ai2 = dict([(0,  0),
                              (1, -1),
                              (2,  1),                             
                              (3,  0),   
                              ])
    
    directions = [[1,0],[0,1],[-1,0],[0,-1]]

    def __init__(self):

        self.x_body = [[random.choice(range(WIDTH)), random.choice(range(HEIGHT))]]
        
        self.v_key = random.choice(range(4))
        self.v =  self.directions[self.v_key]  # 뱀의 속도 
        
    def move(self):
        for i in range(len( self.x_body)-1,0,-1 ):
            self.x_body[i] = self.x_body[i-1]    
            
        
        self.x_body[0] = self.x_body[0]*1    # 머리의 주소를 새로 만들기
        self.x_body[0][0] += self.v[0]   # 속도 만큼 이동시키기
        self.x_body[0][1] += self.v[1]               
        
        
    def change_direction_human(self,event_key): 
        
        #self.v_key = (self.v_key + self.key_direction_human2[event_key]) % 4
        #self.v = self.directions[self.v_key]
        
        # 사람이 할때는 상대좌표계 대신 절대좌표계로
        tmp_v = self.key_direction_human[event_key]               # 방향키 입력 받으면 속도 변화
        
        # 진행방향의 반대방향으로 키입력 눌렀을때, 자기 몸에 겹쳐서 죽는거 방지
        if len(self.x_body) > 1:
            if self.x_body[0][0] + tmp_v[0] != self.x_body[1][0] and \
                   self.x_body[0][1] + tmp_v[1] != self.x_body[1][1]:
               self.v = tmp_v
        else : 
            self.v = tmp_v


    def change_direction_ai(self, ai_action):   # ai_action은 0,1,2,3,4,5 다섯가지 행동 (멈춤, 위, 아래, 왼쪽, 오른쪽)        
                       
        self.v_key = (self.v_key + self.key_direction_ai2[ai_action]) % 4   # 방향키 입력 받으면 속도 변화
        self.v = self.directions[self.v_key]
        
    def draw(self, screen):
        # 뱀을 그린다.
        for x in self.x_body:
            draw_block_game_coor(screen, self.color, x)
        

class Apple:
    """사과 클래스"""
    color = color_apple  # 사과 색깔

    def __init__(self):
        
        random_x = random.choice(range(WIDTH))
        random_y = random.choice(range(HEIGHT))
        self.x = [random_x, random_y]  # 사과의 위치
        
    def draw(self, screen):
        # 사과를 그린다.
        draw_block_game_coor(screen, self.color, self.x)
        
    #사과를 놓고 싶은 위치에 놓는다. x는 길이 2 짜리 리스트
    def _set_x(self, x ):   
        self.x = x   



        

class env_snake:
    """게임판 클래스"""
    width  = WIDTH    # 게임판의 너비
    height = HEIGHT    # 게임판의 높이

    def __init__(self):                
        
        self.snake = Snake()  # 게임판 위의 뱀
        self.apple = Apple()  # 게임판 위의 사과        
        self.eat_count = 0 
        
        self.rewards = 0        
        self.done = False
        
        self.remain_moves = snake_move_increase*3        
        
        self.observation = self.get_observation()
        self.q_score = np.array([[0,0,0]])
        
        self._set_displaying()       
        
       
    def restart_episode(self, train_mode = True):        
        self.snake = Snake()  # 게임판 위의 뱀
        self.apple = Apple()  # 게임판 위의 사과
        self.eat_count = 0
        self.rewards = 0        
        self.done = False
        
        self.remain_moves = snake_move_increase*3                
     
        return True
        
        
    # 사과 먹혔나 참거짓 반환
    def is_apple_eaten(self):        
        return self.snake.x_body[0] == self.apple.x 
    
    # 사과 먹혔을 때 판 위에 아무곳에나 놓기
    def set_apple_again(self):
        
        self.apple._set_x(x = [-1, -1])          # 일단 사과 판 밖에다 치워놓고       
        
        switch_1 = 1
        while switch_1:                          # 뱀 몸에 안겹치는 곳에 사과 놓
            switch_1 = 0
            random_x = random.choice(range(self.width))
            random_y = random.choice(range(self.height))
            
            for snake_body in self.snake.x_body:
                if [random_x, random_y] == snake_body:
                    switch_1 = 1
                    break
        
        self.apple._set_x(x = [random_x, random_y])
        
    # 뱀이 사과 먹었으면 뱀 성장
    def grow_snake(self):
        self.snake.x_body.append(self.snake.x_body[0])        
        
    # 뱀 충돌시 예외 발생시키기
    def collision_check(self):        
        # 자기 몸에 부딪혔을 때 
        if self.snake.x_body[0] in self.snake.x_body[1:-1]:                        
            self.end()
            
        # 벽에 부딪혔을 때
        if self.snake.x_body[0][0] < 0 or self.snake.x_body[0][0] >= self.width \
            or self.snake.x_body[0][1] < 0 or self.snake.x_body[0][1] >= self.height:             
            self.end()
        
        # 이동횟수 제한 초과했을 때
        if self.remain_moves <0 :
            self.done = True
            
    def end(self):
        self.rewards = -100
        self.done = True

    def get_observation(self):          # 싸이즈 16개짜리
                
        ob = self.get_sight()        
        
        return ob, self.done, self.rewards, self.eat_count
    
    def _f_rotation(self, position, v_key):
        # position은 [x,y] 형태로 들어와야함
        if v_key == 0:
            return position
        elif v_key == 1:
            return [position[1], -position[0]]
        elif v_key == 2:
            return [-position[0], -position[1]]
        elif v_key == 3:
            return [-position[1], position[0]]
        else :
            print('something wrong with _f_rotation. check this please')
        
    
    def get_sight(self):          # 싸이즈 8개짜리                
        a = self.apple.x
        b = self.snake.x_body        
        
        # 오른쪽 장애물
        obstacle_right = self.width - b[0][0] 
        for body in self.snake.x_body[1:] :
            tmp_distance = body[0] - b[0][0]
            if body[1] == b[0][1] and tmp_distance > 0 and tmp_distance < obstacle_right:
                obstacle_right = tmp_distance        
        # 아래쪽 장애물
        obstacle_down = self.height - b[0][1]        
        for body in self.snake.x_body[1:] :
            tmp_distance = body[1] - b[0][1]
            if body[0] == b[0][0] and tmp_distance > 0 and tmp_distance < obstacle_down:
                obstacle_down = tmp_distance                
        # 왼쪽 장애물
        obstacle_left = b[0][0] + 1
        for body in self.snake.x_body[1:] :
            tmp_distance =  b[0][0] - body[0] 
            if body[1] == b[0][1] and tmp_distance > 0 and tmp_distance < obstacle_left:
                obstacle_left = tmp_distance        
        # 위쪽 장애물
        obstacle_up = b[0][1] + 1        
        for body in self.snake.x_body[1:] :
            tmp_distance =  b[0][1] - body[1] 
            if body[0] == b[0][0] and tmp_distance > 0 and tmp_distance < obstacle_up:
                obstacle_up = tmp_distance
                
        apple_relative = self._f_rotation([a[0] - b[0][0], a[1] - b[0][1]], self.snake.v_key)
        obstacles = [obstacle_right, obstacle_down, obstacle_left, obstacle_up]
        # 현재 속도 기준으로 좌표 회전
        for _ in range(self.snake.v_key):
            obstacles.append(obstacles.pop(0))
            
        ob = [ apple_relative[0], apple_relative[1] , self.snake.v[0], self.snake.v[1] ]
        ob.extend(obstacles)
              
        
        
        # 뱀머리 둘러싼 8방향 장애물 여부 체크
        ob_cross = [0]*8
        val_1 = 5
        for body in self.snake.x_body[1:] :
            if body[0] - b[0][0] ==  1 and body[1] - b[0][1] ==  0:
                ob_cross[0] = val_1   # 오른쪽
            if body[0] - b[0][0] ==  0 and body[1] - b[0][1] ==  1:
                ob_cross[1] = val_1   # 아래쪽
            if body[0] - b[0][0] == -1 and body[1] - b[0][1] ==  0:
                ob_cross[2] = val_1   # 왼쪽
            if body[0] - b[0][0] ==  0 and body[1] - b[0][1] == -1:
                ob_cross[3] = val_1   # 위쪽
            
            if body[0] - b[0][0] ==  1 and body[1] - b[0][1] == -1:
                ob_cross[4] = val_1   # 오른쪽 위
            if body[0] - b[0][0] ==  1 and body[1] - b[0][1] ==  1:
                ob_cross[5] = val_1   # 오른쪽 아래
            if body[0] - b[0][0] == -1 and body[1] - b[0][1] ==  1:
                ob_cross[6] = val_1   # 왼쪽 아래
            if body[0] - b[0][0] == -1 and body[1] - b[0][1] == -1:
                ob_cross[7] = val_1   # 왼쪽 위
            
            
            
        if b[0][0] == self.width-1:            
            ob_cross[0] = val_1;
            ob_cross[4] = val_1; ob_cross[5] = val_1;
        if b[0][0] == self.height-1:
            ob_cross[1] = val_1;        
            ob_cross[5] = val_1; ob_cross[6] = val_1; 
        if b[0][0] == 0:
            ob_cross[2] = val_1;
            ob_cross[6] = val_1; ob_cross[7] = val_1;            
        if b[0][1] == 0:
            ob_cross[3] = val_1;
            ob_cross[7] = val_1; ob_cross[4] = val_1;
            
        # 현재 속도 기준으로 좌표 회전
        for _ in range(self.snake.v_key):
            ob_cross[:4] = ob_cross[1:4] + ob_cross[0:1]
            ob_cross[4:8] = ob_cross[5:8] + ob_cross[4:5]
            
        ob.extend(ob_cross)  # 총 16개 원소
                
        return ob



    


    ####################### 여기부터 실행하는 함수
        
    def play_ai(self, action):            # 액션 입력받을때만 실행되도록        

        self._move_snake(action)        
        self._decide_next_turn()        
        
        self._display()
        
        
    # 행동 입력 받아서 뱀 움직이고, 상황에 맞게 보상 부여
    def _move_snake(self, action):
        
        # 한 턴당 시작 보상 0
        self.rewards = 0
        
        # 이동
        self.snake.change_direction_ai(action)    # 이동 명령 받기        
        ## 뱀 이동방향이 사과쪽인지 체크 -> 보상        
        if True:
            if self.snake.v[0] * (self.apple.x[0] - self.snake.x_body[0][0]) + \
                  self.snake.v[1] * (self.apple.x[1] - self.snake.x_body[0][1]) >= 0:
                self.rewards += 1
            else:
                self.rewards -= 5
        else :
            self.rewards -= 5  # 불안한 상황에 있다면 이동할때마다 보상 -1
        
        self.snake.move()                
        
        # 사과 먹었는지 체크 -> 먹었으면 점수 추가
        if self.is_apple_eaten():   
            self.rewards += 20    # 사과 먹으면 5점 추가
            self.eat_count += 1
            self.remain_moves += snake_move_increase  # 이동 가능 횟수 추가 
            self.remain_moves = min(self.remain_moves, snake_move_increase*5) # 최대 이동 가능횟수 500으로 제한            
            
            # 불 번쩍번쩍
            L1.on(amp=150+min(2*(self.eat_count),105), rate = 3, switch=True)
            L2.on(amp=150,rate = 50, switch = True)
            L3.on(amp=250,rate = 2, switch = True)
            
            self.apple.color = apple_color_random()
            
        # 충돌 체크
        self.collision_check()
        
        # 현재 관측 내용
        self.observation = self.get_observation()    # 게임 재시작 하기 전에 현재 observation 저장해놓기           
        self.remain_moves -= 1
    

    # 사과 먹었으면 사과 재배치, done = True면 에피소드 재시작
    def _decide_next_turn(self):
        # 사과 먹었는지 체크 -> 먹었으면 사과 재배치
        if self.is_apple_eaten():
            self.set_apple_again()
            self.grow_snake()
            
        if self.done == True:
            self.restart_episode()      
            
# 여기서부터 그림그리기 함수들
    def _set_displaying(self):
                
        # 0 : pygame 초기화
        pygame.init()     
        self.screen = pygame.display.set_mode((SCREEN_WIDTH+SCREEN_INFO_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Snake Babe')  # 파이게임 윈도우창의 글자 변경
        
        # 글자 입력 세팅
        self.font = pygame.font.Font('freesansbold.ttf', 64) 
        self.font2 = pygame.font.Font('freesansbold.ttf', 16) 
        self.font3 = pygame.font.Font('freesansbold.ttf', 16) 
        
    # pygame : event 받고 처리하는 영역 
    def _display(self):
        # 이렇게 해야 파이게임이 돌아간다.
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                exit()


        # screen을 통해 그리기
        draw_background(self.screen)                  
        self._draw(self.screen)  #  화면에 게임상황을 그린다
        pygame.display.update()
        
    # 뱀이랑 사과 그리기
    def _draw(self, screen):        
        self.snake.draw(screen)
        self.apple.draw(screen)
        
        self._draw_texts(screen)
        self._draw_info_window(screen)
        self._draw_info2(screen)
        
        
        # 얘네들을 아래에 놓아서, 사과먹고 한프레임 뒤에 빛이 터진다. 이게 더 예쁨
        global color_background
        color_background = (L1.light(),L1.light(),L1.light())
        L2_light = L2.light()        
        self.snake.color = (30 ,200,100 + L2_light)
        #self.snake.color = (30-L2_light//5 ,200-L2_light//2,100 + L2_light) # 더 화려한거
        
        
    def _draw_texts(self, screen):
        # 사과 먹은 개수 표시
        text = self.font.render('{0}'.format(self.eat_count), True, (150,10,220)) 
        textRect = text.get_rect() 
        textRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2) 
        
        screen.blit(text, textRect)
        
        
    def _draw_info_window(self, screen):
        
        draw_bg_info(screen)
        
        # 조이스틱 표시 
        joystick_position = [[SCREEN_WIDTH + 60, info_joystick_height],[SCREEN_WIDTH + 30, info_joystick_height+30],[SCREEN_WIDTH + 90, info_joystick_height+30]]
        bar_position = [[SCREEN_WIDTH + 30, 350], [SCREEN_WIDTH + 30 + 22, 350],[SCREEN_WIDTH + 30 + 44, 350]] # x좌표 22씩 쓴다.
        action = np.argmax(self.q_score[0])
        for i in range(3):
            if i == action:
                color = [100,200,200]  # 가치 제일 높은 방향 색깔 
            elif self.q_score[0][i] <0 :
                color = [220,100,100] # 가치가 (-)인  방향 색깔
            else : 
                color = [150,150,100]
            draw_block(screen, color, joystick_position[i], size = 30)
            
            # 밑에 각 숫자별 막대그래프 표시
            draw_bar(screen, color, bar_position[i], size_wh =( 20, (self.q_score[0][i]) ) )
            
            
        
        # q_score 적어넣기
        text_position = [[SCREEN_WIDTH + 75, info_joystick_height+15],                        
                         [SCREEN_WIDTH + 45, info_joystick_height+45],
                         [SCREEN_WIDTH + 105, info_joystick_height+45]]
        for i in range(3):
            text = self.font2.render('{0:.0f}'.format(self.q_score[0][i]), True, (0,0,0)) 
            textRect = text.get_rect() 
            textRect.center = (text_position[i][0], text_position[i][1])     
            screen.blit(text, textRect)
        
        # 막대그래프에 글자 표시
        text = self.font3.render('U   L   R     ', True, (0,0,50)) 
        textRect = text.get_rect() 
        textRect.center = (SCREEN_WIDTH + 74, 365)     
        screen.blit(text, textRect)
        
    def _draw_info2(self, screen):
        ob = self.observation[0]*1
        ob[0] = ob[0]*3+120
        ob[1] = ob[1]*3+120
        ob[2] = (ob[2]*10)+120
        ob[3] = (ob[3]*10)+120
        
        ob[4] = ob[4]*3+120
        ob[5] = ob[5]*3+120
        ob[6] = ob[6]*3+120
        ob[7] = ob[7]*3+120
        
        SW = SCREEN_WIDTH + 40
        SH = 210
        ob_position = [[SW, SH],[SW+22, SH],[SW+44, SH],[SW + 66, SH],
                       [SW, SH+30],[SW+22, SH+30],[SW+44, SH+30],[SW + 66, SH+30]]
        for j,i in enumerate(ob_position):            
            pygame.gfxdraw.filled_circle(screen, i[0], i[1], 5, ( min(  50+int(ob[j]*0.2),255) ,int(ob[j]*0.7) ,min(50+ int(ob[j]* 0.6 ),255 ) ) )
            
            
        # 먹으면 반짝
        L3_light = [int(L3.light()*0.9), 
                    min(int(L3.light()*(0.4 + self.eat_count/150 ) ),255), 
                    min(int(L3.light()* (0.7+ self.eat_count/150  )  ) , 255)  ] #
        pygame.gfxdraw.filled_circle(screen, SCREEN_WIDTH + 15, 15, 5, L3_light )
        pygame.gfxdraw.filled_circle(screen, SCREEN_WIDTH + SCREEN_INFO_WIDTH - 15, 15, 5, L3_light )

        
    def close(self):
        
        pygame.quit()
        
    def play_human(self):

        # 1 : event 받고 처리하는 영역
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                exit()
                    
            if event.type == pygame.KEYDOWN:   # 방향 바꾸기                
                self.snake.change_direction_human(event.key)
                    
        self.snake.move()        
        
        if self.is_apple_eaten():
            self.set_apple_again()
            self.grow_snake()
            self.eat_count += 1
            
            L1.on(amp=150+min(2*(self.eat_count),105), rate = 3, switch=True)
            L2.on(amp=150,rate = 50, switch = True)
            L3.on(amp=250,rate = 2, switch = True)            
            
        self.collision_check()
        
        if self.done == True :
            self.restart_episode()
            
        
        # 3 : screen을 통해 그리기
        draw_background(self.screen)                  
        self._draw(self.screen)  
        pygame.display.update()




def play_human():

    
    env = env_snake()
    env.restart_episode()
    
    while True:
        env.play_human()
        time.sleep(0.05)

