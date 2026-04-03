import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import numpy as np


from model_dqn import agent, brain, replay_memory


class Agent:
    def __init__(self, num_actions, batch_size, gamma, replay_memory, model, optimizer ):
        '''태스크의 상태 및 행동의 가짓수를 설정'''
        self.brain = brain.Brain( num_actions, batch_size, gamma, replay_memory, model, optimizer )  # 에이전트의 행동을 결정할 두뇌 역할 객체를 생성

    def update_q_function(self):
        '''Q함수를 수정'''
        self.brain.replay()

    def get_action(self, state, episode):
        '''행동을 결정'''
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        '''memory 객체에 state, action, state_next, reward 내용을 저장'''
        self.brain.memory.push(state, action, state_next, reward)


