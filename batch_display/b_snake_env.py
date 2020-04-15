
#%

RED = 225, 50, 10        # 적색:   적 255, 녹   0, 청   0
GREEN = 30, 200, 100      # 녹색:   적   0, 녹 255, 청   0
BLUE = 0, 0, 255       # 청색:   적   0, 녹   0, 청 255
PURPLE = 127, 0, 127   # 보라색: 적 127, 녹   0, 청 127
BLACK = 0, 0, 0        # 검은색: 적   0, 녹   0, 청   0
GRAY = 127, 127, 127   # 회색:   적 127, 녹 127, 청 127
WHITE = 255, 255, 255  # 하얀색: 적 255, 녹 255, 청 255

color_snake = GREEN * 1
color_apple = RED * 1
color_background = BLACK * 1



import pygame             
import time

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

# 뱀 이동 횟수

snake_move_increase = (WIDTH + HEIGHT)*2



def draw_background(screen):
    """게임의 배경을 그린다."""
    background = pygame.Rect((0, 0), (B_WIDTH, B_HEIGHT))
    pygame.draw.rect(screen, color_background, background)      # 그림을 그릴땐 screen에 그려야 한다.

def draw_block(screen, color, position ):
    """position 위치에 color 색깔의 블록을 그린다."""
    block = pygame.Rect((position[0] * BLOCK_SIZE, position[1] * BLOCK_SIZE),
                        (BLOCK_SIZE, BLOCK_SIZE))
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
    
    key_direction_ai = dict([(0, [ 0,-1]),
                             (1, [ 0, 1]),
                             (2, [-1, 0]),
                             (3, [ 1, 0]),
                             (4, [ 0, 0])
                             ])

    def __init__(self):
        self.x_body = [[9, 6], [9, 7], [9, 8], [9, 9]]  # 뱀의 위치
        self.v = [1, 0]  # 뱀의 속도 (초기값 [0,1])
        
    def move(self):
        for i in range(len( self.x_body)-1,0,-1 ):
            self.x_body[i] = self.x_body[i-1]    ### *1 안하면 값 대신 주소를 받아와서 엉망되는것 주의해야해!!!!
            
        
        self.x_body[0] = self.x_body[0]*1    # 머리의 주소를 새로 만들기
        self.x_body[0][0] += self.v[0]   # 속도 만큼 이동시키기
        self.x_body[0][1] += self.v[1]               
        
        
    def change_direction_human(self,event_key): 
        tmp_v = self.key_direction_human[event_key]               # 방향키 입력 받으면 속도 변화
        
        # 진행방향의 반대방향으로 키입력 눌렀을때, 자기 몸에 겹쳐서 죽는거 방지
        if self.x_body[0][0] + tmp_v[0] != self.x_body[1][0] and \
               self.x_body[0][1] + tmp_v[1] != self.x_body[1][1]:
           self.v = tmp_v


    def change_direction_ai(self, ai_action):   # ai_action은 0,1,2,3,4,5 다섯가지 행동 (멈춤, 위, 아래, 왼쪽, 오른쪽)        
        tmp_v = self.key_direction_ai[ai_action]               # 방향키 입력 받으면 속도 변화
        
        # 진행방향의 반대방향으로 키입력 누르지 않는건 ai가 스스로 배워야해        
        if self.x_body[0][0] + tmp_v[0] != self.x_body[1][0] and \
               self.x_body[0][1] + tmp_v[1] != self.x_body[1][1]:
           self.v = tmp_v
           
        #self.v = self.key_direction_ai[ai_action]               # 방향키 입력 받으면 속도 변화
        
    def draw(self, screen):
        # 뱀을 그린다.
        for x in self.x_body:
            draw_block(screen, self.color, x)
            
    def b_draw(self, screen, shift):        
        for x in self.x_body:
            shifted_position = x*1
            shifted_position[0] += shift[0]
            shifted_position[1] += shift[1]
            draw_block(screen, self.color, shifted_position)
        

class Apple:
    """사과 클래스"""
    color = color_apple  # 사과 색깔

    def __init__(self):
        
        random_x = random.choice(range(WIDTH))
        random_y = random.choice(range(HEIGHT))
        self.x = [random_x, random_y]  # 사과의 위치
        
    def draw(self, screen):
        # 사과를 그린다.
        draw_block(screen, self.color, self.x)
        
    def b_draw(self, screen, shift):
        
        shifted_position = self.x * 1
        shifted_position[0] += shift[0]
        shifted_position[1] += shift[1]        
        draw_block(screen, self.color, shifted_position)
        
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
        #self.past_ob = self.get_sight()  #  지난번 움직임 기억하기 위한 설정
        self.observation = self.get_observation()
        
        #self._set_displaying()       
        
       
    def restart_episode(self, train_mode = True):        
        self.snake = Snake()  # 게임판 위의 뱀
        self.apple = Apple()  # 게임판 위의 사과
        self.eat_count = 0
        self.rewards = 0        
        self.done = False
        
        self.remain_moves = snake_move_increase*3
        
        #self.past_ob = self.get_sight()
     
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
        self.rewards -= 100
        self.done = True

    def get_observation(self):          # 싸이즈 16개짜리
                
        ob = self.get_sight()        
        #ob.extend(self.past_ob)        
        
        return ob, self.done, self.rewards, self.eat_count
    
    def get_sight(self):          # 싸이즈 8개짜리        
        a = self.apple.x
        b = self.snake.x_body        
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
        ob = [ a[0] - b[0][0], a[1] - b[0][1] , self.snake.v[0], self.snake.v[1] ,
              obstacle_left, obstacle_up, obstacle_right, obstacle_down  ]
        
        
        # 뱀머리 둘러싼 8방향 장애물 여부 체크
        ob_cross = [0]*8
        val_1 = 5
        for body in self.snake.x_body[1:] :
            if body[0] - b[0][0] == -1 and body[1] - b[0][1] == -1:
                ob_cross[0] = val_1
            if body[0] - b[0][0] == -1 and body[1] - b[0][1] == 1:
                ob_cross[1] = val_1
            if body[0] - b[0][0] == 1 and body[1] - b[0][1] == -1:
                ob_cross[2] = val_1
            if body[0] - b[0][0] == 1 and body[1] - b[0][1] == 1:
                ob_cross[3] = val_1
            
            if body[0] - b[0][0] == -1 and body[1] - b[0][1] == 0:
                ob_cross[4] = val_1
            if body[0] - b[0][0] ==  0 and body[1] - b[0][1] == -1:
                ob_cross[5] = val_1
            if body[0] - b[0][0] ==  1 and body[1] - b[0][1] == 0:
                ob_cross[6] = val_1
            if body[0] - b[0][0] ==  0 and body[1] - b[0][1] == 1:
                ob_cross[7] = val_1
                
        if b[0][0] == 0:
            ob_cross[0] = val_1; ob_cross[1] = val_1;
            ob_cross[4] = val_1;
        if b[0][1] == 0:
            ob_cross[0] = val_1; ob_cross[2] = val_1;
            ob_cross[5] = val_1;
        if b[0][0] == self.width-1:
            ob_cross[2] = val_1; ob_cross[3] = val_1;
            ob_cross[6] = val_1;
        if b[0][0] == self.height-1:
            ob_cross[1] = val_1; ob_cross[3] = val_1;
            ob_cross[7] = val_1;
        ob.extend(ob_cross)
                
        return ob
    


####################### 여기부터 실행하는 함수
        
    def play_ai(self, action):            # 액션 입력받을때만 실행되도록        

        self._move_snake(action)        
        self._decide_next_turn()        
        
        #self._display()
        
        
    # 행동 입력 받아서 뱀 움직이고, 상황에 맞게 보상 부여
    def _move_snake(self, action):
        
        # 한 턴당 시작 보상 0
        self.rewards = 0
        
        # 이동
        self.snake.change_direction_ai(action)    # 이동 명령 받기        
        ## 뱀 이동방향이 사과쪽인지 체크 -> 보상
        if self.snake.v[0] * (self.apple.x[0] - self.snake.x_body[0][0]) + \
              self.snake.v[1] * (self.apple.x[1] - self.snake.x_body[0][1]) >0:
            self.rewards += 1
        else:
            self.rewards -= 5
        self.snake.move()        
        self.rewards -= 0.1      # 이동 할 때 마다 0.1씩 감점
        
        # 사과 먹었는지 체크 -> 먹었으면 점수 추가
        if self.is_apple_eaten():   
            self.rewards += 20    # 사과 먹으면 5점 추가
            self.eat_count += 1
            self.remain_moves += snake_move_increase  # 이동 가능 횟수 추가 
            self.remain_moves = min(self.remain_moves, snake_move_increase*5) # 최대 이동 가능횟수 500으로 제한            
            
        # 충돌 체크
        self.collision_check()
        
        # 현재 관측 내용
        self.observation = self.get_observation()    # 게임 재시작 하기 전에 현재 observation 저장해놓기           
        self.remain_moves -= 1
    

    # 사과 먹었으면 사과 재배치, done = True면 에피소드 재시
    def _decide_next_turn(self):
        # 사과 먹었는지 체크 -> 먹었으면 사과 재배치
        if self.is_apple_eaten():
            self.set_apple_again()
            self.grow_snake()
            
        if self.done == True:
            self.restart_episode()      
            



#%%

class batch_env_snake:
    def __init__(self,N):        
        
        self.N = N
        self.envs = []        
        for i in range(N):
            self.envs.append(env_snake())            
        
        self.observation = []
        for i in range(len(self.envs[0].observation)):   # self.observation의 차원을 (원래 차원, 배치차원) 으로 한다
            self.observation.append( [env.observation[i] for env in self.envs] )
            
        self._set_displaying()
            
    def restart_episode(self):
        
        for env in self.envs:
            env.restart_episode()
            
    def play_ai(self, action_b):
        
        assert len(action_b) == self.N
        
        for action, env in zip(action_b, self.envs):
            env.play_ai(action)
            
        
        for i in range(len(self.envs[0].observation)):  
            self.observation[i] = [env.observation[i] for env in self.envs] 
            
        self.display()


# 여기부터 그림그리기 함수들
    def _set_displaying(self):
        
        
        global BLOCK_SIZE,WIDTH,HEIGHT,SCREEN_WIDTH,SCREEN_HEIGHT, B_WIDTH, B_HEIGHT
        
        BLOCK_SIZE = 5
        WIDTH = 25
        HEIGHT = 25
        
        # 그리기용 변수 
        SCREEN_WIDTH = BLOCK_SIZE * WIDTH     # 화면에 4xn 크기로 그릴거야 
        SCREEN_HEIGHT = BLOCK_SIZE * HEIGHT 
        
        B_WIDTH = SCREEN_WIDTH * 4 
        B_HEIGHT = SCREEN_HEIGHT  * ( (self.N+1)//4)
        
                
        # 0 : pygame 초기화
        pygame.init()     
        self.screen = pygame.display.set_mode((B_WIDTH, B_HEIGHT))
        pygame.display.set_caption('Snake Babes')  # 파이게임 윈도우창의 글자 변경
        
        # 글자 입력 세팅
        self.font = pygame.font.Font('freesansbold.ttf', 16) 
        
    # pygame : event 받고 처리하는 영역 
    def display(self):
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
        
        for i in range(self.N):            
            #print([ i%4 * SCREEN_WIDTH, i//4 * SCREEN_HEIGHT ], end = ' ')
            self.envs[i].snake.b_draw(screen, [ i%4 * WIDTH, i//4 * HEIGHT ])
            self.envs[i].apple.b_draw(screen, [ i%4 * WIDTH, i//4 * HEIGHT ]) 
            #self.envs[i].snake.draw(screen)
            #self.envs[i].apple.draw(screen)
        #self._draw_texts(screen)

        
    def _draw_texts(self, screen):
        # 사과 먹은 개수 표시
        text = self.font.render('{0}'.format(self.eat_count), True, (150,10,220)) 
        textRect = text.get_rect() 
        textRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2) 
        
        screen.blit(text, textRect)
        
    
       

        
    def close(self):
        
        pygame.quit()
    
            
        
#%%



        
##################################

# 예외 새로 만들면 이렇게 만드나봐
class SnakeCollisionException(Exception):
    def __init__(self):
        super().__init__('에러메시지')
        print('hey')
        #self.play_human()
        
    pass

#########################################