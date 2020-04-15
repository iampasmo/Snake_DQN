#%%
from snake_env import env_snake, play_human


#%%
import numpy as np
import random

from collections import deque

import pickle
import copy

import time
import os

#% 신경망
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.layers.merge import add

from keras.models import Model, load_model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras import backend as K



#%%

class model_for_agent:
    def __init__(self, name = None):
        
        self._build()
        self._compile()
        self.name = name
        
        
        return
    
    # 인풋을 사과 2개 + 뱀 몸통하나당 2개 * 몸통 100개로 하자
    def _build(self):
        self.input = Input( shape=(16,), name = 'Input')
        
        x = self.input
        x = Dense(units = 48)(x)
        x = LeakyReLU()(x)
        x = Dense(units = 48)(x)
        x = LeakyReLU()(x)
        x = Dropout(rate = 0.1)(x)
        x = Dense(units = 64)(x)
        x = LeakyReLU()(x)
        x = Dense(units = 24)(x)
        x = LeakyReLU()(x)
        # Duel DQN   # 층 얇을때 학습 더 잘되는 경향 있어서 Advantage부분 한층 줄임
        L1 = Dense(units = 24)(x)             
        A1 = LeakyReLU()(L1)
        Advantage = Dense(units = 3)(A1)
        # Critic
        V1 = Dense(units =24)(L1)
        Value = Dense(units = 1)(V1) 
        
        def duel_q_out(args):
            Value, Advantage = args
            return Value + (Advantage - K.mean(Advantage, axis = 1, keepdims = True))
        
        x = Lambda(duel_q_out)([Value, Advantage])
        
        
        self.output = x        
        self.model = Model(self.input, self.output)
        
        
    def _compile(self):
        opt = Adam(lr = 0.0005)
        self.model.compile(loss = 'mse', optimizer = opt, metrics = ['accuracy'])
        
    def train(self, x_train, y_train):        
        return self.model.train_on_batch(x_train, y_train)
    
    
        



#%%
class DQNAgent():
    def __init__(self):
        
        # 클래스의 함수들을 위한 값 설정 
        self.q_model = model_for_agent("Q")
        self.target_model = model_for_agent("target")

        self.memory = deque(maxlen=mem_maxlen)          

        self.epsilon = epsilon_init

        self.q_score = [[0,0,0]]

    # Epsilon greedy 기법에 따라 행동 결정
    def get_action(self, state):
        if self.epsilon > np.random.rand():
            # 랜덤하게 행동 결정
            return np.random.randint(0, action_size)
        else:
            state = np.reshape(state,(1,-1))   
            self.q_score = self.q_model.model.predict(state)  # 신경망 거친 후에 각 행동별 예상 가치
            predict = np.argmax( self.q_score )            
            
            return np.asscalar(predict)

    # 리플레이 메모리에 데이터 추가 (상태, 행동, 보상, 다음 상태, 게임 종료 여부)
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 네트워크 모델 저장 
    def save_model(self, number):
        # save 하는 함수 만들어야해
        self.target_model.model.save("target_model_{0}.h5".format(number))
        return

    # 학습 수행 
    def train_model(self, done):
        
        # Epsilon 값 감소 
        if done:
            if self.epsilon > epsilon_min:
                self.epsilon -= epsilon_d_rate / (run_episode - start_train_episode)
                


        # 학습을 위한 미니 배치 데이터 샘플링
        mini_batch = random.sample(self.memory, batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in range(batch_size):
            states.append(mini_batch[i][0])
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states.append(mini_batch[i][3])
            dones.append(mini_batch[i][4])
      
        # 타겟값 계산 
        
        states = np.array(states)
        next_states = np.array(next_states)
        
        
        
        target = self.q_model.model.predict(states)
        
        target_val = self.target_model.model.predict(next_states) # original dqn
        
        # double dqn
        arg_target_val = np.argmax(self.q_model.model.predict(next_states), axis = 1)    # q_model의 선택 
        
        
        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i] 
            else:
                #target[i][actions[i]] = rewards[i] + discount_factor * np.amax(target_val[i]) # original dqn
                target[i][actions[i]] = rewards[i] + discount_factor * target_val[i][arg_target_val[i]] # q_model이 선택한 행동의 target_model에서의 값으로 업데이트
                
            
        
        # 학습 수행 및 손실함수 값 계산        
        loss = self.q_model.train(x_train = states, y_train = target)
      
        return loss

    # 타겟 네트워크 업데이트 
    def update_target(self):        
        self.target_model.model.set_weights(self.q_model.model.get_weights())

#%% 게임 돌릴 하이퍼 파라미터



# DQN을 위한 파라미터 값 세팅 

action_size = 3

train_mode = True

batch_size = 32
mem_maxlen = 50000
discount_factor = 0.9
learning_rate = 0.0025

run_episode = 25000
test_episode = 1000

start_train_episode =  100

target_update_step = 500
print_interval = 20
save_interval = 1000

epsilon_init = 0.4
epsilon_min = 0.01
epsilon_d_rate = 15


pause_time = 0.0    

#%%
    
video_tape = []


#%%

# Main 함수 -> 전체적으로 DQN 알고리즘을 진행 
def play_ai(epochs_trained = 3000, load_model_switch = True):
    
    global action_size,batch_size,discount_factor,epsilon_d_rate,epsilon_init,epsilon_min,learning_rate,mem_maxlen,print_interval,run_episode,save_interval,start_train_episode,target_update_step,test_episode,train_mode,pause_time
    
    
    # 게임 환경 불러오기
    env = env_snake()

    # DQNAgent 클래스를 agent로 정의 
    agent = DQNAgent()
    
    
    # 학습모드일때, 아닐때 구분
    if load_model_switch:
        
        epsilon_init = 0; epsilon_min = 0;        
        mem_maxlen = 50000
        
        start_train_episode = 0;  run_episode = 0 ;  test_episode = 100        
        print_interval = 20
        agent.epsilon = 0        
        
        pause_time = 0.03
        
        model_path = os.path.join("trial 1","target_model_{}.h5".format(epochs_trained))
        agent.q_model.model.set_weights(load_model(model_path).get_weights())
        agent.target_model.model.set_weights(load_model(model_path).get_weights())
    
    step = 0
    rewards = []
    losses = [] 
    apples = []
    
    env.restart_episode()

    # 게임 진행 반복문     
    for episode in range(run_episode + test_episode):    
    
        if episode > run_episode:
            train_mode = False
        
        # 상태, episode_rewards, done 초기화 
        env.observation = env.get_observation() # 이거 해줘야 초기화됨!!
        state = env.observation[0]
        episode_rewards = 0
        done = False

        # 한 에피소드를 진행하는 반복문 
        while not done:
            step += 1

            # 행동 결정 및 게임 환경에 행동 적용 
            action = agent.get_action(state)                 # 에이전트의 신경망을 통해 행동 받아옴
            ## q_score 넘겨주기
            env.q_score = agent.q_score
            ### 플레이하면 녹화해놓기
            video_tape.append(copy.deepcopy([state, action, agent.q_score, env.snake.x_body, env.apple.x]))
            
            env.play_ai(action)                              # env에 행동 입력

            # 다음 상태, 보상, 게임 종료 정보 취득 
            next_state = env.observation[0]                  # 이자리에 뱀이랑 사과 위치가 와야 하고
            reward = env.observation[2]                      # 이 위치에 보상이 와야해
            episode_rewards += reward
            done = env.observation[1]                        # 이 위치에 에피소드 끝났는지 여부 들어와야해
            
            
            # 학습 모드인 경우 리플레이 메모리에 데이터 저장 
            agent.append_sample(state, action, reward, next_state, done)
            
                
                

            # 상태 정보 업데이트 
            state = next_state*1

            if episode > start_train_episode and train_mode:
                # 학습 수행 
                loss = agent.train_model(done)
                losses.append(loss)

                # 타겟 네트워크 업데이트 
                if step % (target_update_step) == 0:
                    agent.update_target()
                    
            time.sleep(pause_time)    # 학습 안할때는 느리게
                    
        rewards.append(episode_rewards)
        apples.append(env.observation[3])

        # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록 
        if episode % print_interval == 0 and episode != 0:
            print("step: {} / episode: {} / reward: {:.2f} / loss: {:.4f} / epsilon: {:.3f} / apples : {:.1f}".format
                  (step, episode, np.mean(rewards), np.mean(losses), agent.epsilon, np.mean(apples)))
            
            rewards = []
            losses = []
            apples = []
            

        # 네트워크 모델 저장 
        if episode % save_interval == 0 and episode != 0:
            agent.save_model(episode)
            print("Save Model {}".format(episode))
            

    with open('agent_memory.pkl','wb') as f:
        pickle.dump( agent.memory, f )
        
    with open('video_tape.pkl','wb') as f:
        pickle.dump( video_tape, f )
    
    
    
    env.close()

#%%
