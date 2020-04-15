#%%
from b_snake_env import batch_env_snake

#%%

# 사과 가까이 가는 보상 5->1, 멀리가는 보상 -10 -> -2
# 눈 여덟개 달았음

#%%
import numpy as np
import random
import datetime
import time
import tensorflow as tf
from collections import deque

import pickle


#% 신경망
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.layers.merge import add
#from models.layers.layers import ReflectionPadding2D
from keras.models import Sequential, Model, load_model
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
        # Duel DQN
        L1 = Dense(units = 24)(x)             
        A1 = LeakyReLU()(L1)
        Advantage = Dense(units = 4)(A1)
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
class b_DQNAgent():
    def __init__(self):
        
        # 클래스의 함수들을 위한 값 설정 
        self.q_model = model_for_agent("Q")
        self.target_model = model_for_agent("target")

        self.memory = deque(maxlen=mem_maxlen)          

        self.epsilon = epsilon_init



    # Epsilon greedy 기법에 따라 행동 결정
    def b_get_action(self, state):
        if self.epsilon > np.random.rand():
            # 랜덤하게 행동 결정
            return np.random.randint(0, action_size, size = (N,) )
        else:
            state = np.reshape(state,(N,-1))            
            predict = np.argmax( self.q_model.model.predict(state), axis = 1 )
            
            return predict

    # 리플레이 메모리에 데이터 추가 (상태, 행동, 보상, 다음 상태, 게임 종료 여부)
    def append_sample(self, state, action, reward, next_state, done):
        #self.memory.append((state, action, reward, next_state, done))
        for i in range(N):
            self.memory.append((state[i], action[i], reward[i], next_state[i], done[i]))

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
        arg_target_val = np.argmax(self.q_model.model.predict(next_states), axis = 1)         
        
        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                #target[i][actions[i]] = rewards[i] + discount_factor * np.amax(target_val[i]) # original dqn
                target[i][actions[i]] = rewards[i] + discount_factor * target_val[i][arg_target_val[i]]
                
        # 학습 수행 및 손실함수 값 계산 
       
        loss = self.q_model.train(x_train = states, y_train = target)
      
        return loss

    # 타겟 네트워크 업데이트 
    def update_target(self):        
        self.target_model.model.set_weights(self.q_model.model.get_weights())
# =============================================================================
#         
#         for i in range(len(self.model.trainable_var)):
#             self.sess.run(self.target_model.trainable_var[i].assign(self.model.trainable_var[i]))
# =============================================================================
# =============================================================================
# 
#     # 텐서보드에 기록할 값 설정 및 데이터 기록 
#     def Make_Summary(self):
#         self.summary_loss = tf.placeholder(dtype=tf.float32)
#         self.summary_reward = tf.placeholder(dtype=tf.float32)
#         tf.summary.scalar("loss", self.summary_loss)
#         tf.summary.scalar("reward", self.summary_reward)
#         Summary = tf.summary.FileWriter(logdir=save_path, graph=self.sess.graph)
#         Merge = tf.summary.merge_all()
# 
#         return Summary, Merge
#     
#     def Write_Summray(self, reward, loss, episode):
#         self.Summary.add_summary(
#             self.sess.run(self.Merge, feed_dict={self.summary_loss: loss, 
#                                                  self.summary_reward: reward}), episode)
#     
# =============================================================================
#%% 게임 돌릴 하이퍼 파라미터



# DQN을 위한 파라미터 값 세팅 

action_size = 4

train_mode = True

batch_size = 32
mem_maxlen = 50000
discount_factor = 0.9
learning_rate = 0.0025

run_episode = 25000
test_episode = 1000

start_train_episode =  1000 

target_update_step = 500
print_interval = 100
save_interval = 1000

epsilon_init = 0.4
epsilon_min = 0.01
epsilon_d_rate = 15


load_target_model_switch = False
if load_target_model_switch:      
    epsilon_init = 0; epsilon_min = 0;
    start_train_episode = 5000; learning_rate = 0.001
    
    mem_maxlen = 50000
    
    start_train_episode =  0
    run_episode = 0
    test_episode = 500
    
    print_interval = 10
    epsilon_test = 0.0
#%% Batch Envs

N = 12 # 동시에 돌릴 환경 개수



#%%

# Main 함수 -> 전체적으로 DQN 알고리즘을 진행 
#if __name__ == '__main__':
def play_ai():
    
    global action_size,batch_size,discount_factor,epsilon_d_rate,epsilon_init,epsilon_min,learning_rate,mem_maxlen,print_interval,run_episode,save_interval,start_train_episode,target_update_step,test_episode,train_mode
    
    #env = env_snake()
    env = batch_env_snake(N)
    
    # DQNAgent 클래스를 agent로 정의 
    agent = b_DQNAgent()
    
    
    if load_target_model_switch:
        agent.q_model.model.set_weights(load_model("target_model_7000.h5").get_weights())
        agent.target_model.model.set_weights(load_model("target_model_7000.h5").get_weights())
    
    step = 0
    rewards = []
    losses = [] 
    apples = []
    
    env.restart_episode()

    # 게임 진행 반복문     
    for episode in range(run_episode + test_episode):
    #for episode in range(1):
    
        if episode > run_episode:
            train_mode = False
        
        # 상태, episode_rewards, done 초기화 
        state = env.observation[0]
        episode_rewards = 0
        done = [False]*N

        # 한 에피소드를 진행하는 반복문 
        while not done[0]:
            step += 1

            # 행동 결정 및 유니티 환경에 행동 적용 
            action = agent.b_get_action(state)                 # 에이전트의 신경망을 통해 행동 받아옴
            env.play_ai(action)                              # env에 행동 입력

            # 다음 상태, 보상, 게임 종료 정보 취득 
            next_state = env.observation[0]                  # 이자리에 뱀이랑 사과 위치가 와야 하고
            reward = env.observation[2]                      # 이 위치에 보상이 와야해
            episode_rewards += reward[0]
            done = env.observation[1]                        # 이 위치에 에피소드 끝났는지 여부 들어와야해
            
            
           
            # 학습 모드인 경우 리플레이 메모리에 데이터 저장 
            if train_mode:
                agent.append_sample(state, action, reward, next_state, done)
            else:
                #time.sleep(0.01) 
                agent.append_sample(state, action, reward, next_state, done)
                agent.epsilon = epsilon_test

            # 상태 정보 업데이트 
            state = next_state*1

            if episode > start_train_episode and train_mode:
                # 학습 수행 
                loss = agent.train_model(done)
                losses.append(loss)

                # 타겟 네트워크 업데이트 
                if step % (target_update_step) == 0:
                    agent.update_target()
                    
            #time.sleep(0.03)
                    
        rewards.append(episode_rewards)
        apples.append(env.observation[3])

        # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록 
        if episode % print_interval == 0 and episode != 0:
            print("step: {} / episode: {} / reward: {:.2f} / loss: {:.4f} / epsilon: {:.3f} / apples : {:.1f}".format
                  (step, episode, np.mean(rewards), np.mean(losses), agent.epsilon, np.mean(apples)))
            #agent.Write_Summray(np.mean(rewards), np.mean(losses), episode)
            rewards = []
            losses = []
            apples = []
            

        # 네트워크 모델 저장 
        if episode % save_interval == 0 and episode != 0:
            agent.save_model(episode)
            print("Save Model {}".format(episode))
            

    with open('agent_memory.pkl','wb') as f:
        pickle.dump( agent.memory, f )
    
    
    
    env.close()
        #print('hello')

    #env.close()
#%%
