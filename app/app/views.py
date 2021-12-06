from flask import request
from flask_restx import Resource
from app import api
import logging
import pandas as pd
import json
import os
from celery import Celery
import pymongo        

CELERY_BROKER = os.environ['CELERY_BROKER']
CELERY_BACKEND = os.environ['CELERY_BACKEND']

celery = Celery('tasks', broker=CELERY_BROKER, backend=CELERY_BACKEND)

log = logging.getLogger(__name__)

def receive_result(task_id):
    res = celery.AsyncResult(task_id)
    while res.state == 'PENDING':
        res = celery.AsyncResult(task_id)
    else:
        return res.result
    
def try_first_connection():
    print('connect')
    try:
        task = celery.send_task('get_params', args = [0, 'count'])
        res = receive_result(task.id)
        return True
    except Exception as e:
        return True
    
check = try_first_connection()

@api.route('/api/ml_models')
class MLModels(Resource):

    def get(self):
        user_id = eval(request.form.get('user_id'))
        log.info('Get all models')
        task = celery.send_task('get_params', args = [user_id, 'ml_models'])
        res = receive_result(task.id)
        return str(res)

    def delete(self):
        user_id = eval(request.form.get('user_id'))
        log.info('Clear all models')
        ml_models = []
        del_dict = {'ml_models':ml_models, 'counter':0}
        task = celery.send_task('update_params', args = [user_id, del_dict])
        res = receive_result(task.id)
        
@api.route('/api/ml_models/can_train')
class MLModels(Resource):

    def get(self):
        log.info('Get list of all models API can train')
        task = celery.send_task('can_train')
        res = receive_result(task.id)
        return res


@api.route('/api/ml_models/count')
class MLModels(Resource):

    def get(self):
        user_id = eval(request.form.get('user_id'))
        log.info('Get all trained models count')
        task = celery.send_task('get_params', args = [user_id, 'counter'])
        res = receive_result(task.id)
        return str(res)


@api.route('/api/ml_models/<int:id>')
class MLModel(Resource):

    def get(self, id):
        log.info('Get model')
        log.info(f'id = {id}\n type(id) = {type(id)}')
        user_id = eval(request.form.get('user_id'))
        
        task = celery.send_task('get', args = [user_id, id])
        res = receive_result(task.id)
        return res

    def put(self, id):
        log.info('Retrain model')
        log.info(f'id = {id}\n type(id) = {type(id)}')
        
        try:
            user_id = eval(request.form.get('user_id'))
            train = request.form.get('train')
            y_train = request.form.get('y_train')
            test = request.form.get('test')
            y_test = request.form.get('y_test')
            name = request.form.get('name')
            params = request.form.get('params')
            cv = request.form.get('cv')
            if cv is None:
                cv = 3
            else:
                cv = eval(cv)
            
            if params is None:
                params = {}
            else:
                params = eval(params)

            assert train is not None
            assert y_train is not None

            find_params = request.form.get('find_params')
            if find_params is not None:
                config = request.form.get('config')
                if config is None:
                    config = {}
                else:
                    config = eval(config)

                n_trials = request.form.get('n_trials')
                if n_trials is None:
                    n_trials = 30
                else:
                    n_trials = eval(n_trials)

            assert train is not None
            assert y_train is not None

            if find_params:
                log.info('Find hyperparameters')
                task1 = celery.send_task('find_hyperparams', args = [name, train, y_train, n_trials, cv, config])
                res = receive_result(task1.id)
                params = res
            
            task = celery.send_task('retrain', args = [user_id, id, train, y_train, test, y_test, params, cv])
            r = receive_result(task.id)
            return r

        except IndexError as e:
            api.abort(404, e)
        except Exception as e:
            api.abort(404, e)
            
    def delete(self, id):
        log.info('Delete model')
        log.info(f'id = {id}\n type(id) = {type(id)}')
        user_id = eval(request.form.get('user_id'))
        
        task = celery.send_task('delete_one', args = [user_id, id])
        r = receive_result(task.id)
        
            
@api.route('/api/ml_models/train')
class MLModel(Resource):

    def post(self):
        log.info('Train new model')
        
        try:
            user_id = eval(request.form.get('user_id'))
        
            train = request.form.get('train')
            y_train = request.form.get('y_train')
            
            test = request.form.get('test')
            y_test = request.form.get('y_test')
            name = request.form.get('name')
            params = request.form.get('params')
            cv = request.form.get('cv')
            if cv is None:
                cv = 3
            else:
                cv = eval(cv)

            if params is None:
                params = {}
            else:
                params = eval(params)

            find_params = request.form.get('find_params')
            if find_params is not None:
                config = request.form.get('config')
                if config is None:
                    config = {}
                else:
                    config = eval(config)

                n_trials = request.form.get('n_trials')
                if n_trials is None:
                    n_trials = 30
                else:
                    n_trials = eval(n_trials)

            assert train is not None
            assert y_train is not None

            if find_params:
                log.info('Find hyperparameters')
                task1 = celery.send_task('find_hyperparams', args = [name, train, y_train, n_trials, cv, config])
                res = receive_result(task1.id)
                params = res
            
            task = celery.send_task('train', args = [user_id, name, train, y_train, test, y_test, params, cv])
            r = receive_result(task.id)
            
            return r
        except KeyError as e:
            api.abort(404, e)
        except Exception as e:
            api.abort(404, e)


@api.route('/api/ml_models/predict/<int:id>')
class MLModel(Resource):

    def get(self, id):
        log.info('Predict')
        log.info(f'id = {id}\n type(id) = {type(id)}')
        user_id = eval(request.form.get('user_id'))
        
        test = request.form.get('test')
        y_test = request.form.get('y_test')

        task = celery.send_task('predict', args = [user_id, id, test, y_test])
        r = receive_result(task.id)

        return r
        
