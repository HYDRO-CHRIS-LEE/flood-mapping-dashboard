# -----------------------------------------------------------------------------------------------
#  Flappybird_helper.py
# -----------------------------------------------------------------------------------------------

import numpy as np

import gym

import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import os
import shutil
from os import walk
import itertools
import statistics
import time
import requests
import json
import warnings
import random
from datetime import datetime
import re

from model_dqn import agent, brain, replay_memory
import env_flappybird.flappybird_env as f_env

import sys


# 저장 경로
MODELS_PATH = 'saved_models'

# 훈련 중 best model 저장 경로
MODELS_ON_TRAINING_PATH = 'saved_models/best_on_training'

# 로그 경로
LOG_PATH = 'runs'

# Min upload interval (변경 금지)
MIN_UPLOAD_INTERVAL = 5

# 수평선
LINE_STR = '-' * 80

# 로딩시 불러올 파일 갯수
NUM_LISTING_MODELS = 10

# Competition에서의 max step
MAX_STEP_COUNT = 1000000


class FlappyBirdHelper():

    def __init__( self, app, nickname, auth_code, num_ep_train, num_attempts_tune, num_ep_tune, num_ep_eval, is_random_gap, gap_size, render_skip, save_best_model, save_on_tuning ):

        # flappybird url
        self.FB_URL = 'https://45d75z9xcc.execute-api.ap-northeast-2.amazonaws.com/live/ycs-admin'

        # TSP Version
        self.FB_HELPER_VERSION = '2021.11.20v1'

        self.app = app

        self.student_id = ''
        self.class_id = ''
        self.semester = ''
        self.sname = ''
        self.nickname = nickname
        self.auth_code = auth_code

        self.num_ep_train = num_ep_train
        self.num_attempts_tune = num_attempts_tune
        self.num_ep_tune = num_ep_tune
        self.num_ep_eval = num_ep_eval

        self.is_random_gap = is_random_gap
        self.gap_size = gap_size

        self.render_skip = render_skip
        self.save_best_model = save_best_model
        self.save_on_tuning = save_on_tuning

        self.env = f_env.FlappyBirdEnv()

        self.current_episode = 0
        self.max_steps = 0
        self.uploaded_time = 0

        self.render_counter = 0



    # 메뉴 출력
    def menu(self):

        prompt = '\n{0}\nFlappyBird Competition 2021-2\n{0}\n'.format( LINE_STR )
        prompt += '(H) Hyperparameter Tuning\n'
        prompt += '(T) Train from scratch and Save\n'
        prompt += '(L) Load and Continue Training\n'
        prompt += '(E) Load and Evaluate\n'
        prompt += '(J) Load and Join Competition\n'
        prompt += '(C) Clear Temp Files (saved_models/best_on_training/*.pt)\n'
        prompt += '(D) Delete All Models Except Top 10\n'
        prompt += '(P) Print Model\'s Parameters\n'
        prompt += '(Q) Quit\n'
        prompt += '{0}\nSelect Menu > '.format( LINE_STR )

        while True:            
            m = input( prompt )
            m = m.lower()

            if m == 'h':
                self.onselect_tune()
            elif m == 'l':
                loaded_model_info = self._load()
                if loaded_model_info:
                    self.onselect_train(loaded_model_info=loaded_model_info)
            elif m == 't':
                self.onselect_train()
            elif m == 'e':
                loaded_model_info = self._load()
                if loaded_model_info:
                    self.onselect_eval(loaded_model_info=loaded_model_info)
            elif m == 'j':
                loaded_model_info = self._load()
                if loaded_model_info:
                    self.onselect_comp(loaded_model_info=loaded_model_info)
            elif m == 'c':
                self.delete_all_files(MODELS_ON_TRAINING_PATH)

            elif m == 'd':
                self.onselect_remove_except_10_models()

            elif m == 'p':
                loaded_model_info = self._load()
                if loaded_model_info:
                    self.onselect_print_model(loaded_model_info=loaded_model_info)

            elif m == 'q':
                break



    # 모델 저장하기
    def save_model(self, filename, best_so_far=False ):

        models_path = MODELS_PATH

        if best_so_far:
            models_path = MODELS_ON_TRAINING_PATH

        # 디렉토리가 없는 경우 생성
        if os.path.exists(models_path) == False:
            os.mkdir(models_path)

        pt_fname = models_path + '/' + filename + '.pt'
        
        torch.save( self.agent.brain.model, pt_fname )

       
        print('{0}\nTrained model saved! --->>> {1}'.format( LINE_STR, filename + '.pt' ) )


    def delete_all_files( self, folder ):

        if not os.path.exists(folder):
            return

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))        

        print('All files are deleted ({}/*.*)'.format(folder))


    def _load(self):

        model_list = []
        for filename in os.listdir(MODELS_PATH):

            if filename.endswith('.pt'):

                mlist = re.match( r'avg ([\d.]+) max ([\d.]+) ', filename )
                mlist2 = re.match( r'score ([\d.]+) avg ([\d.]+) max ([\d.]+) ', filename )

                if mlist:
                    model_list.append( { 'filename': filename, 
                    'score': 0,
                    'avg_score': mlist[1],
                    'max_score': mlist[2] } )
                elif mlist2:
                    model_list.append( { 'filename': filename, 
                    'score': mlist2[1],
                    'avg_score': mlist2[2],
                    'max_score': mlist2[3] } )                   

        if model_list:
            try:
                sorted_by_score = sorted( model_list, key=lambda model: float(model['score']), reverse=True )
            except:
                sorted_by_score = model_list
        else:
            print('Error: No saved models!')
            return None

        sorted_list = sorted_by_score
        
        sorted_list = sorted_list[:NUM_LISTING_MODELS]

        while True:

            print('\n{0}\nLoad a model\n{0}'.format('-'*80))

            for i, item in enumerate(sorted_list):
                print( '({}) {}'.format( i, item['filename'] ) )
            
            print('(C) Cancel')
            print('{}'.format('-'*80))
                
            m = input('Enter a model # to load > ')

            if m.lower() == 'c':
                return None

            try:
                m = int(m)
            except:
                print('Wrong number.')
                continue

            if 0 <= m < len(sorted_list):
                filepath = MODELS_PATH + '/' + sorted_list[m]['filename']
                loaded_model_info = {}
                loaded_model_info['model'] = torch.load( filepath )
                loaded_model_info['score'] = sorted_list[m]['score']
                loaded_model_info['avg_score'] = sorted_list[m]['avg_score']

                print('{0}\nTrained model loaded! --->>> {1}'.format( LINE_STR, sorted_list[m]['filename'] ) )

                return loaded_model_info

        return None


    def onselect_train(self, loaded_model_info=None):

        now = datetime.now() # current date and time
        date_str = now.strftime("%Y-%m-%d-%H.%M.%S")

        if loaded_model_info:
            self.agent = self.app.init_agent( self.env, model_to_use=loaded_model_info['model'], random_hyperparameters=False )
        else:
            self.agent = self.app.init_agent( self.env, random_hyperparameters=False )

        self._train( self.num_ep_train, train_str = 'Train', model_id=self.app.model_id, date_str=date_str )


    # 하이퍼파라미터 최적화
    def onselect_tune(self):

        now = datetime.now() # current date and time
        date_str = now.strftime("%Y-%m-%d-%H.%M.%S")

        for trial_index in range(1, self.num_attempts_tune+1):
            # random hyperparameter를 사용하도록 Agent/Brain 재생성
            self.agent = self.app.init_agent( self.env, random_hyperparameters=True )
            self._train( self.num_ep_tune, train_str = 'Tune', date_str=date_str, trial_index=trial_index, num_trials=self.num_attempts_tune )


    # 모델 평가
    def onselect_eval(self, loaded_model_info=None):

        if loaded_model_info:
            self.agent = self.app.init_agent( self.env, model_to_use=loaded_model_info['model'], random_hyperparameters=False )
        else:
            print('Error: No loaded model.')
            return

        self.agent.brain.model.eval()

        score_list = []
        
        for episode in range( self.num_ep_eval ):

            self.current_episode = episode + 1

            observation = self.env.reset(is_random_gap = self.is_random_gap, gap_size= self.gap_size)

            state = observation
            state = torch.from_numpy(state).type( torch.FloatTensor )
            state = torch.unsqueeze(state, 0)

            for step in itertools.count(0):

                with torch.no_grad():
                    action = self.agent.brain.model(state).argmax(1).view(1,1)

                observation_next, reward, done, _ = self.env.step( action.item(), gap_size= self.gap_size )

                self.env.render( self.render_skip )

                score = self.env.score

                if done == False:  
                    state_next = observation_next  # 관측 결과를 그대로 상태로 사용
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)  # numpy 변수를 파이토치 텐서로 변환
                    state_next = torch.unsqueeze(state_next, 0)  # size 4를 size 1*4로 변환

                reward = torch.FloatTensor([reward])

                # 관측 결과를 업데이트
                state = state_next

                if done:  

                    score_list.append(score)

                    avg_score = sum(score_list) / len(score_list)
                    max_score = max(score_list)

                    print('[Eval] episode {} - score: {:6.2f} avg: {:6.2f} max: {:3d}'.format(self.current_episode, score, avg_score, max_score ) )
                    break


    def onselect_print_model(self, loaded_model_info):

        content = ''
        
        if loaded_model_info:
            for name, param in loaded_model_info['model'].named_parameters():
                if param.requires_grad:
                    content += '[ {} ]\n{}\n\n'.format( name, str(param.data) )

            print( '\n\n{}\n{}'.format( loaded_model_info['model'], content) )

    def onselect_remove_except_10_models(self):

        model_list = []
        for filename in os.listdir( MODELS_PATH ):
            if filename.endswith('.pt'):

                mlist = re.match( r'avg ([\d.]+) max ([\d.]+) ', filename )
                mlist2 = re.match( r'score ([\d.]+) avg ([\d.]+) max ([\d.]+) ', filename )

                if mlist:
                    model_list.append( { 'filename': filename, 
                    'score': 0,
                    'avg_score': mlist[1],
                    'max_score': mlist[2] } )
                elif mlist2:
                    model_list.append( { 'filename': filename, 
                    'score': mlist2[1],
                    'avg_score': mlist2[2],
                    'max_score': mlist2[3] } )                   

        if model_list:
            try:
                sorted_by_score = sorted( model_list, key=lambda model: float(model['score']), reverse=True )
            except:
                sorted_by_score = model_list
        else:
            print('Error: No saved models!')
            return None

        sorted_list = sorted_by_score
        
        sorted_list = sorted_list[10:]


        for i, item in enumerate(sorted_list):

            file_path = MODELS_PATH + '/' + item['filename']

            try:
                os.unlink(file_path)
                print( 'Remove {}...'.format(file_path) )

            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))        


    # 모델 훈련
    def _train(self, num_episodes, train_str='Train', model_id=None, date_str='', trial_index=1, num_trials=1 ):

        self.logs = []
        
        self.current_episode = 0
        self.max_steps = 0
        self.max_score = 0

        best_steps = 0
        best_counter = 0

        max_score = 0.0
        avg_score = 0.0
        avg_steps = 0.0

        steps_list = []
        score_list = []

        for episode in range(num_episodes):

            self.current_episode = episode + 1

            observation = self.env.reset(is_random_gap = self.is_random_gap, gap_size=self.gap_size)

            state = observation
            state = torch.from_numpy(state).type( torch.FloatTensor )
            state = torch.unsqueeze(state, 0)

            for step in itertools.count(0):

                action = self.agent.get_action(state, episode)  # 다음 행동을 결정

                observation_next, reward, done, _ = self.env.step( action.item(), gap_size=self.gap_size  )

                score = self.env.score

                self.env.render( self.render_skip )

                if done == False:  
                    state_next = observation_next  # 관측 결과를 그대로 상태로 사용
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)  # numpy 변수를 파이토치 텐서로 변환
                    state_next = torch.unsqueeze(state_next, 0)  # size 4를 size 1*4로 변환

                reward = torch.FloatTensor([reward])

                self.agent.memorize(state, action, state_next, reward)

                self.agent.update_q_function()

                # 관측 결과를 업데이트
                state = state_next

                if done:
                    steps_list.append( step )
                    score_list.append( score )

                    train_label = '[{train_str}] Trial ({trial_index:2d}/{num_trials:3d})  Episode ({episode:3d}/{num_episodes:3d})'.format(train_str=train_str, trial_index=trial_index, num_trials=num_trials, episode=episode, num_episodes= num_episodes)
                    train_label += ' Score: {}'.format( score )

                    print(train_label)

                    self.logs.append( ( score, episode) )

                    break

            # Save the best model so far
            if step > best_steps and self.save_best_model and (train_str == 'Train' or self.save_on_tuning):

                max_score = max(score_list)
                avg_score = sum(score_list[-30:]) / len(score_list[-30:])

                filename = 'score {:.2f} avg {:.2f} max {}'.format( score, avg_score, max_score )
                filename += ' {} {} ({})'.format( date_str, train_str, best_counter )

                best_steps = step
                self.save_model( filename, True )

                best_counter += 1


        max_score = max(score_list)
        avg_score = sum(score_list[-30:]) / len(score_list[-30:])

        # Save the last model
        if train_str == 'Train' or self.save_on_tuning:

            filename = 'score {:.2f} avg {:.2f} max {}'.format( score, avg_score, max_score )
            filename += ' {} {}'.format( date_str, train_str )

            self.save_model( filename )

        # Write on Tensorboard
        writer = SummaryWriter( '{}/{}-{}'.format( LOG_PATH, train_str, date_str ) )


        for log in self.logs:
            writer.add_scalar( '{}-{}/trial_index_{}'.format(train_str, date_str, trial_index), log[0], log[1] )

        writer.add_hparams( hparam_dict={ 'trial_index': trial_index, 'model': self.app.model_id, 'optimizer':self.app.optimizer_id,
            'lr': self.app.learning_rate, 'bsize':self.app.batch_size, 'dropout':self.app.dropout_rate, 'gamma':self.app.gamma, 'replay_mem': self.app.replay_memory_capacity },
            metric_dict={ 'score': score_list[-1], 'avg_score': avg_score, 'max_score': max_score  } )

        writer.close()


    
    # for the competition
    def onselect_comp(self, loaded_model_info=None):

        if loaded_model_info:
            self.agent = self.app.init_agent( self.env, model_to_use=loaded_model_info['model'], random_hyperparameters=False )
        else:
            print('Error: No loaded model.')
            return

        comp_rounds = self.download_from_server()

        if len(comp_rounds) == 0:
            print('No open competitions.')
            return

        print( '\n----- Competition Rounds -----\n')
        print('      ({}) {}'.format(0, 'All rounds'))
        for i, comp_round in enumerate(comp_rounds):
            print( '      ({}) {}'.format( i+1, comp_round.get('round_id', 'error') ) )
        
        selected = input('\nSelect a number > ')

        print('\n')

        self.agent.brain.model.eval()

        for i, comp_round in enumerate(comp_rounds):

            round_id = comp_round.get('round_id')
            episode_num = int( comp_round.get('episode_num') )
            score_limit = int( comp_round.get('score_limit') )
            gap_size = int( comp_round.get('gap_size') )
            is_random_gap = comp_round.get('is_random_gap') == 'true'

            if selected == '0' or selected == str(i+1):
                print('\n*** Evaluation <{}> ***'.format(round_id))
            else:
                continue

            competition_scores = []

            for episode in range(episode_num):

                self.current_episode = episode

                observation = self.env.reset( is_random_gap = is_random_gap, gap_size = gap_size )

                state = observation
                state = torch.from_numpy(state).type( torch.FloatTensor )
                state = torch.unsqueeze(state, 0)

                for step in range(MAX_STEP_COUNT):

                    with torch.no_grad():
                        action = self.agent.brain.model(state).argmax(1).view(1,1)

                    observation_next, reward, done, _ = self.env.step( action.item(), gap_size = gap_size )

                    self.env.render( self.render_skip )

                    if done == False:  
                        state_next = observation_next  # 관측 결과를 그대로 상태로 사용
                        state_next = torch.from_numpy(state_next).type(torch.FloatTensor)  # numpy 변수를 파이토치 텐서로 변환
                        state_next = torch.unsqueeze(state_next, 0)  # size 4를 size 1*4로 변환

                    reward = torch.FloatTensor([reward])

                    # 관측 결과를 업데이트
                    state = state_next

                    if self.env.score >= score_limit:
                        done = True

                    if done:
                        competition_scores.append( self.env.score )
                        print('<{}> episode {}/{} score: {}'.format( round_id, episode+1, episode_num, self.env.score ) ) 
                        break

                if done == False:
                    competition_scores.append( self.env.score )
                    print('<{}> episode {}/{} score: {} - max score '.format( round_id, episode+1, episode_num, self.env.score ) )

            print('*** Evaluation <{}>  score avg {:.2f} max {} \n'.format( round_id, statistics.mean(competition_scores), max(competition_scores) ) )
            self.uploaded_time = time.time() - MIN_UPLOAD_INTERVAL * 2
            self.submit_result(round_id, competition_scores )


    # check upload time
    def check_upload_time(self):

        current_time = time.time()

        if current_time >= self.uploaded_time + MIN_UPLOAD_INTERVAL:
            self.uploaded_time = current_time
            return True

        return False


    # download from the competition server - DO NOT MODIFY
    def download_from_server(self):

        params = { 'get_rounds': 1 }
        data = { 'auth_code': self.auth_code, 'comp_type': 'flappybird' }
        
        r = requests.post( self.FB_URL, params=params, data=data )

        current_time = time.time()

        try:

            if r.status_code == 200:

                response = r.json()

                if response.get('result') == 'success':
                    rounds = response.get('rounds')

                    print('{} rounds are downloaded'.format( len(rounds)) )

                    open_rounds = []

                    for rd in rounds:
                        st = int( rd.get('start_timestamp') )
                        et = int( rd.get('end_timestamp') )

                        # if current_time >= st and current_time <= et:
                        open_rounds.append(rd)

                    return open_rounds

                else:
                    print('{}'.format(response.get('message')))
            else:
                print('ERROR({}): Could not download from the FB server'.format(r.status_code))
                sys.exit()
                
        except Exception as e:
            print('Exception: ' + e)
            sys.exit()


    # upload to the competition server - DO NOT MODIFY
    def submit_result(self, round_id, scores):

        if self.check_upload_time() == False:
            return

        avg_score = statistics.mean( scores )
        max_score = max( scores )

        params = { 'flappybird_submit_result': 1 }
        data = { 'round_id':round_id, 'auth_code': self.auth_code, 'nickname': self.nickname, 'avg_score': avg_score, 'max_score': max_score, 'score_list': json.dumps(scores) }

        r = requests.post( self.FB_URL, params=params, data=data )

        try:

            if r.status_code == 200:

                response = r.json()

                if response.get('result') == 'success':

                    if response.get('valid') == 'valid':
                        print('<{}> result uploaded.'.format(round_id) )
                else:
                    print( "CLOSED: {}".format( response.get('error') ) )

            else:
                print('Server error({})'.format(r.status_code))

        except Exception as e:
            print(len(r.text))
            print('Exception: ', e)
            sys.exit()

