# -----------------------------------------------------------------------------------------------
# (Lab) Flappybird_app_lab.py - version 2021.11.20v1 
# -----------------------------------------------------------------------------------------------

import numpy as np
import torch
from torch import nn
from torch import optim

import os
from os import walk
import itertools
import statistics
import time
import requests
import json
import warnings
import random

import gym
from model_dqn import agent, brain, replay_memory

#-------------------------------------------------------
# 간단한 설명서
#-------------------------------------------------------
#
# - 코드 실행: python flappybird_app_lab.py
# - 프롬프트에서 이전 명령어 다시 불러오기: 위쪽 방향키 누름
# - 실행 도중 멈추기: Ctrl + C
# - 종료: 쉘 윈도우를 닫거나 메뉴에서 q 입력
# - 정상 스피드 플레이: 게임 화면 포커스 후 왼쪽 방향키 누름
#
# - 훈련/튜닝 도중 저장되는 모델의 확장자는 (.pt) 이며 위치는 아래와 같음
# -     에피소드 끝나고 저장된 모델: ./saved_models
# -     에피소드 중간에 저장된 모델: ./saved_models/best_on_training
# - evaluation이나 competition 참여를 위해서 사용하려는 후보 모델은 ./saved_models 로 복사하면 리스트에 뜸
#       만약 모델이 너무 많은 경우 좋은 후보 모델만 남기고 지우거나 다른 임시 폴더에 옮겨놓으면 됨
#
# - TensorBoard 에서 불러오는 로그: ./runs
#       만약 로그가 너무 많이 쌓였을 경우 ./runs 안의 폴더를 선택하여 지우면 됨
#
# - 헬퍼가 버전 업 되었을 경우: flappybird_helper.py 를 지우면 새버전 자동 다운로드
#
# *** 제출할 때 Competition에 사용할 모델 하나를 선택하여 이름에 competition을 추가하여 다른 모델과 구분하기 바람
#     예) score 0.00 avg 0.00 max 0 2021-11-20-17.24.15 Tune competition.pt
#
# *** 제출할 때 저장된 모델과 로그가 너무 많은 경우 일부 지우고 일부는 남겨두어 본인이 직접 훈련을 진행했음을 입증하는 증거를 남겨 두도록 함
# 
#-------------------------------------------------------


#-------------------------------------------------------
# *** 기본 설정 (변경 가능) ***
#-------------------------------------------------------

NICKNAME = '빅스비' # 표시용 별명

# https://esohn.be/register 의 Auth Code 항목을 복사한 후 붙여넣음
AUTH_CODE = 'ff6e9a9b47614f038ba2e6d93dbf4ce1'  # 개인 인증코드

# 훈련 도중 지금까지 최고기록 모델 저장 여부
SAVE_BEST_MODEL = True

# 하이퍼파라미터 튜닝 때도 모델 저장 여부
SAVE_BEST_MODEL_ON_TUNING = True

# 게임 화면 스킵 프레임 수 (0: 정상속도, 클수록 빠름) (변경 가능)
RENDER_SKIP = 100

LINE_STR = '-' * 120


#-------------------------------------------------------
# *** 하이퍼파라미터 (1) (변경 가능) ***
#-------------------------------------------------------
NUM_EP_EVAL = 400 # 평가 에피소드 수 (변경 가능)
NUM_EP_TRAIN = 10000 # 훈련 에피소드 수 (변경 가능)
NUM_EP_TUNE = 300 # (하이퍼파라미터 튜닝용) 훈련 에피소드 수 (변경 가능)
NUM_ATTEMPTS_TUNE = 100 # 하이퍼파라미터 튜닝 시도 수 (변경 가능)

#-------------------------------------------------------
# *** 하이퍼파라미터 (2) - For Training (변경 가능) ***
#-------------------------------------------------------

# 괜찮은 하이퍼파라미터를 찾았다면 아래에 값을 고정하고 훈련을 더 오래 진행해볼 수 있음

# 학습률
LEARNING_RATE = 0.0001
# 배치사이즈
BATCH_SIZE = 64
# 드롭아웃 비율
DROPOUT = 0.2
# 감가 보상률 - 1에 가까울 수록 미래의 보상을 중요하게 여기게 됨
GAMMA = 0.98
# 리플레이 메모리 용량 - 신경망을 훈련할 때 리플레이 메모리 안에서 랜덤하게 일부를 가져와서 훈련을 진행하는데 이 용량을 설정
REPLAY_MEMORY_CAPACITY = 500000
# 옵티마이저
OPTIMIZER_ID = 'adam'
# 신경망
MODEL_ID = 'model2'

#-------------------------------------------------------
# *** 하이퍼파라미터 (2) - For Tuning (변경 가능) ***
#-------------------------------------------------------

# 어떤 하이퍼파라미터 값이 좋을지 찾기 위해 좀 더 넓은 범위에서 시작해서 범위를 좁혀가며 하이퍼파라미터 튜닝을 진행할 수 있음
# 
# 아래와 같이 적용할 값들을 튜플 안에 지정해 놓으면 랜덤하게 선택해서 적용하게 됨
# 물론 꼭 여러 값을 시도해야 하는 것은 아니며 그럴 경우 튜플 형식은 유지한 채로 하나의 값을 남겨두면 됨:
#     예를들어 옵티마이저를 아담으로만 하기 원하면 RANDOM_OPTIMIZER_ID = ('adam')

# 학습률 범위
RANDOM_LEARNING_RATE_RANGE = ( 0.0001, 0.00005 )
# 배치사이즈 범위
RANDOM_BATCH_SIZE_RANGE = (64,)
# 드롭아웃 비율 범위
RANDOM_DROPOUT_RANGE = ( 0.2, 0.3, 0.4 )
# 감가 보상률 범위
RANDOM_GAMMA_RANGE = ( 0.99, 0.98 )
# 리플레이 메모리 용량 범위
RANDOM_REPLAY_MEMORY_CAPACITY_RANGE = ( 100000, 500000 )
# 옵티마이저 범위
RANDOM_OPTIMIZER_ID = ('adam',)
# 신경망 범위
RANDOM_MODEL_ID = ('model1', 'model2', 'model3')


#-------------------------------------------------------
# *** 하이퍼파라미터 (3) 게임 환경 (변경 가능) ***
#-------------------------------------------------------

# 파이프 갭 랜덤 여부 True/False 이 경우 훈련하는 동안 (120 ~ 170) 범위에서 랜덤하게 파이프 갭이 적용됨
IS_RANDOM_GAP = False

# 파이프 갭 랜덤이 False일 경우 고정 파이프 갭 크기 (120 ~ 170) 범위에서 할당 할 수 있음
GAP_SIZE = 120


# -----------------------------------------------------------------------------------------------
# FlappyBirdApp
# -----------------------------------------------------------------------------------------------
class FlappyBirdApp():

    def __init__(self):

        global AUTH_CODE

        # -----------------------------------------------------------------------------------------------
        # Check Auth Code
        # -----------------------------------------------------------------------------------------------
        if not AUTH_CODE:
            AUTH_CODE = input('Input Auth Code( http://esohn.be/register/ ) > ')
        
        # -----------------------------------------------------------------------------------------------
        # Download FB Helper
        # -----------------------------------------------------------------------------------------------
        if not os.path.isfile('flappybird_helper.py'):
            self.download_fb_helper()


        # Import FB Helper
        import flappybird_helper

        # Create TSP Helper object
        self.helper = flappybird_helper.FlappyBirdHelper( self, nickname=NICKNAME, auth_code=AUTH_CODE,
            num_ep_train=NUM_EP_TRAIN, num_ep_tune=NUM_EP_TUNE, num_ep_eval=NUM_EP_EVAL, num_attempts_tune=NUM_ATTEMPTS_TUNE, 
            is_random_gap=IS_RANDOM_GAP, gap_size=GAP_SIZE, render_skip=RENDER_SKIP, save_best_model=SAVE_BEST_MODEL, save_on_tuning=SAVE_BEST_MODEL_ON_TUNING )

        print( '\n\n{0}  Flappybird Helper Version: {1}  {0}\n\n'.format( '-'*10, self.helper.FB_HELPER_VERSION) )



    def init_agent(self, env, model_to_use=None, random_hyperparameters=False):

        # -----------------------------------------------------------------------------------------------
        # 상태 및 액션 갯수 (변경 금지)
        # -----------------------------------------------------------------------------------------------
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        # -----------------------------------------------------------------------------------------------
        # 하이퍼파라미터 지정
        # -----------------------------------------------------------------------------------------------
        if random_hyperparameters:
            # 적정 범위의 랜덤한 값 지정 (부득이 랜덤이 아닌 순차적 반복을 원할 경우 변경 가능)
            self.learning_rate = random.choice( RANDOM_LEARNING_RATE_RANGE )
            self.batch_size = random.choice( RANDOM_BATCH_SIZE_RANGE )
            self.dropout_rate = random.choice( RANDOM_DROPOUT_RANGE )
            self.gamma = random.choice( RANDOM_GAMMA_RANGE )
            self.replay_memory_capacity = random.choice( RANDOM_REPLAY_MEMORY_CAPACITY_RANGE )
            self.optimizer_id = random.choice( RANDOM_OPTIMIZER_ID )
            self.model_id = random.choice( RANDOM_MODEL_ID )
        else:
            # 튜닝이 끝난 하이퍼파라미터를 집중 훈련
            self.learning_rate = LEARNING_RATE
            self.batch_size = BATCH_SIZE
            self.dropout_rate = DROPOUT
            self.gamma = GAMMA
            self.replay_memory_capacity = REPLAY_MEMORY_CAPACITY
            self.optimizer_id = OPTIMIZER_ID
            self.model_id = MODEL_ID

        # -----------------------------------------------------------------------------------------------
        # *** 신경망 모델 (변경 가능, 여러개의 모델을 만들어서 시도해 볼 수도 있음) ***
        # -----------------------------------------------------------------------------------------------

        model1 = nn.Sequential(
            nn.Linear( num_states, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout( p=self.dropout_rate ),
            
            nn.Linear( 64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout( p=self.dropout_rate ),
            
            nn.Linear( 32, 16 ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout( p=self.dropout_rate ),

            nn.Linear( 16, 8 ),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout( p=self.dropout_rate),
            nn.Linear(8, num_actions),
        )

        model2 = nn.Sequential(
            nn.Linear( num_states, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout( p=self.dropout_rate ),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout( p=self.dropout_rate ),
            
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout( p=self.dropout_rate ),

            nn.Linear( 32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout( p=self.dropout_rate),
            nn.Linear( 8, num_actions),
        )

        model3 = nn.Sequential(
            nn.Linear( num_states, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout( p=self.dropout_rate ),
            
            nn.Linear( 32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout( p=self.dropout_rate ),
            
            nn.Linear( 32, 16 ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout( p=self.dropout_rate ),

            nn.Linear( 16, 16 ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout( p=self.dropout_rate),
            nn.Linear( 16, num_actions),
        )


        # -----------------------------------------------------------------------------------------------
        # 모델 Print 
        # -----------------------------------------------------------------------------------------------

        if model_to_use:
            model = model_to_use
            self.model_id = 'Loaded model'
            print('Using the loaded model...\n\n {}\n{}\n'.format(model, LINE_STR))
        elif self.model_id == 'model1':
            model = model1
            print('Using the model1...\n\n {}\n{}\n'.format(model, LINE_STR))
        elif self.model_id == 'model2':
            model = model2
            print('Using the model2...\n\n {}\n{}\n'.format(model, LINE_STR))
        elif self.model_id == 'model3':
            model = model3
            print('Using the model3...\n\n {}\n{}\n'.format(model, LINE_STR))


        # -----------------------------------------------------------------------------------------------
        # *** Optimizer (변경 가능, Optimizer 추가 등 가능) ***
        # -----------------------------------------------------------------------------------------------

        if self.optimizer_id == 'adam':
            optimizer = optim.Adam( model.parameters(), lr=self.learning_rate )

        elif self.optimizer_id == 'sgd':
            optimizer = optim.SGD( model.parameters(), lr=self.learning_rate )


        # -----------------------------------------------------------------------------------------------
        # Agent 초기화 
        # -----------------------------------------------------------------------------------------------

        memory = replay_memory.ReplayMemory( self.replay_memory_capacity )

        return agent.Agent( num_actions = num_actions, batch_size = self.batch_size, gamma = self.gamma, replay_memory = memory, model = model, optimizer = optimizer )

        # -----------------------------------------------------------------------------------------------


    # -------------------------------------------------------------------------------------------
    # FB Helper download - (변경 금지)
    # -------------------------------------------------------------------------------------------
    def download_fb_helper(self):

        FB_HELPER_URL = 'https://esohn.be/python/flappybird_helper.py'
        
        r = requests.get(FB_HELPER_URL, allow_redirects=True)
       
        if r.status_code == 200:
            with open('flappybird_helper.py', 'wb') as f:
                f.write( r.content )
        else:
            print('ERROR: unable to download fb_helper.py!')



if __name__=="__main__":

    warnings.filterwarnings("ignore")

    app = FlappyBirdApp()

    app.helper.menu()

    app.helper.env.close()



