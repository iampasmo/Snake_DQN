import pygame

import pickle
import time

import numpy as np
#%%





# Main 함수 -> 전체적으로 DQN 알고리즘을 진행 
#if __name__ == '__main__':
def video_player():
    
    #global action_size,batch_size,discount_factor,epsilon_d_rate,epsilon_init,epsilon_min,learning_rate,mem_maxlen,print_interval,run_episode,save_interval,start_train_episode,target_update_step,test_episode,train_mode
    
    
    # 게임 환경 불러오기
    env = env_snake()

    
    
    
    env.restart_episode()
    
    
    
    
    with open('video_tape_from_batch_training.pkl','rb') as f:
        video_tape = pickle.load(f)   
    

    i = 1    
    play_speed = 1
    while True:
        
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                exit()
            
            if event.type == pygame.KEYDOWN:   
                #print(event.key)
                # 재생 방향
                if event.key == 275:
                    play_speed += 1
                elif event.key == 276:
                    play_speed -= 1
                elif event.key == 32:
                    play_speed = 0
                    
                # 멈춰놨을때 프레임 이동
                elif event.key == 273:
                    i += 1
                elif event.key == 274:
                    i -= 1
                    
                # 플레이 속도
                elif event.key == 257: play_speed = 1 * np.sign(play_speed)    
                elif event.key == 258: play_speed = 2 * np.sign(play_speed)
                elif event.key == 259: play_speed = 3 * np.sign(play_speed)
                elif event.key == 260: play_speed = 4 * np.sign(play_speed)    
                elif event.key == 261: play_speed = 5 * np.sign(play_speed)                
                    
        i += play_speed
        
        env.q_score = video_tape[i][2]
        env.snake.x_body = video_tape[i][3]
        env.apple.x = video_tape[i][4]
        
        env._display()
        
        
        time.sleep(0.03)
            
            
                    
            #time.sleep(0.03)
                    
        

        # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록 
        
    
    
    
    env.close()


#%%
    
from snake_env import env_snake
    
video_player()
#%%


