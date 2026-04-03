
import numpy as np

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from . import replay_memory, common

import random


class Brain:

    def __init__(self, num_actions, batch_size, gamma, replay_memory, model, optimizer):

        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma

        self.memory = replay_memory
        self.model = model
        self.optimizer = optimizer


    def replay(self):
        '''Experience Replay로 신경망의 결합 가중치 학습'''

        # -----------------------------------------
        # 1. 저장된 transition 수 확인
        # -----------------------------------------
        # 1.1 저장된 transition의 수가 미니배치 크기보다 작으면 아무 것도 하지 않음
        if len(self.memory) < self.batch_size:
            return

        # -----------------------------------------
        # 2. 미니배치 생성
        # -----------------------------------------
        # 2.1 메모리 객체에서 미니배치를 추출
        transitions = self.memory.sample(self.batch_size)

        # 2.2 각 변수를 미니배치에 맞는 형태로 변형
        # transitions는 각 단계 별로 (state, action, state_next, reward) 형태로 BATCH_SIZE 갯수만큼 저장됨
        # 다시 말해, (state, action, state_next, reward) * BATCH_SIZE 형태가 된다
        # 이것을 미니배치로 만들기 위해
        # (state*BATCH_SIZE, action*BATCH_SIZE, state_next*BATCH_SIZE, reward*BATCH_SIZE) 형태로 변환한다
        batch = common.Transition(*zip(*transitions))

        # 2.3 각 변수의 요소를 미니배치에 맞게 변형하고, 신경망으로 다룰 수 있도록 Variable로 만든다
        # state를 예로 들면, [torch.FloatTensor of size 1*4] 형태의 요소가 BATCH_SIZE 갯수만큼 있는 형태이다
        # 이를 torch.FloatTensor of size BATCH_SIZE*4 형태로 변형한다
        # 상태, 행동, 보상, non_final 상태로 된 미니배치를 나타내는 Variable을 생성
        # cat은 Concatenates(연접)을 의미한다
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        # -----------------------------------------
        # 3. 정답신호로 사용할 Q(s_t, a_t)를 계산
        # -----------------------------------------
        # 3.1 신경망을 추론 모드로 전환
        self.model.eval()

        # 3.2 신경망으로 Q(s_t, a_t)를 계산
        # self.model(state_batch)은 왼쪽, 오른쪽에 대한 Q값을 출력하며
        # [torch.FloatTensor of size BATCH_SIZEx2] 형태이다
        # 여기서부터는 실행한 행동 a_t에 대한 Q값을 계산하므로 action_batch에서 취한 행동 a_t가 
        # 왼쪽이냐 오른쪽이냐에 대한 인덱스를 구하고, 이에 대한 Q값을 gather 메서드로 모아온다
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # 3.3 max{Q(s_t+1, a)}값을 계산한다 이때 다음 상태가 존재하는지에 주의해야 한다

        # cartpole이 done 상태가 아니고, next_state가 존재하는지 확인하는 인덱스 마스크를 만듬
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)))
        # 먼저 전체를 0으로 초기화
        next_state_values = torch.zeros(self.batch_size)

        # 다음 상태가 있는 인덱스에 대한 최대 Q값을 구한다
        # 출력에 접근하여 열방향 최대값(max(1))이 되는 [값, 인덱스]를 구한다
        # 그리고 이 Q값(인덱스=0)을 출력한 다음
        # detach 메서드로 이 값을 꺼내온다
        next_state_values[non_final_mask] = self.model( non_final_next_states ).max(1)[0].detach()

        # 3.4 정답신호로 사용할 Q(s_t, a_t)값을 Q러닝 식으로 계산한다
        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # -----------------------------------------
        # 4. 결합 가중치 수정
        # -----------------------------------------
        # 4.1 신경망을 학습 모드로 전환
        self.model.train()

        # 4.2 손실함수를 계산 (smooth_l1_loss는 Huber 함수)
        # expected_state_action_values은
        # size가 [minibatch]이므로 unsqueeze하여 [minibatch*1]로 만든다
        # loss = F.smooth_l1_loss(state_action_values,
        #                         expected_state_action_values.unsqueeze(1))
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # 4.3 결합 가중치를 수정한다
        self.optimizer.zero_grad()  # 경사를 초기화
        loss.backward()  # 역전파 계산
        self.optimizer.step()  # 결합 가중치 수정


    def decide_action(self, state, episode):
        '''현재 상태에 따라 행동을 결정한다'''

        # ε-greedy 알고리즘에서 서서히 최적행동의 비중을 늘린다
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()  # 신경망을 추론 모드로 전환
            with torch.no_grad():
                # action = self.model(state).max(1)[1].view(1, 1)
                action = self.model(state).argmax(1).view(1,1)
            # 신경망 출력의 최댓값에 대한 인덱스 = max(1)[1]
            # .view(1,1)은 [torch.LongTensor of size 1] 을 size 1*1로 변환하는 역할을 한다

        else:
            # 행동을 무작위로 반환(0 혹은 1)
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])  # 행동을 무작위로 반환(0 혹은 1)
            # action은 [torch.LongTensor of size 1*1] 형태가 된다

        return action
