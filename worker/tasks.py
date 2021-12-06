from celery import Celery
import time
import os
from mlmodels import MLModelsDAO, Objective, users
import pandas as pd
import json
models_dao = MLModelsDAO()

CELERY_BROKER = os.environ['CELERY_BROKER']
CELERY_BACKEND = os.environ['CELERY_BACKEND']
celery = Celery('tasks', broker=CELERY_BROKER, backend=CELERY_BACKEND)

@celery.task(name='get_params')    
def get_params(user_id, param):
    print(user_id)
    return str(models_dao.get_params(user_id = user_id, param = param))

@celery.task(name='update_params')    
def update_params(user_id, params):
    models_dao.update_params(user_id = user_id, params = params)

@celery.task(name='can_train')
def can_train():
    return models_dao.available_models
    
@celery.task(name='get')
def get(user_id, id):
    try:
        return str(models_dao.get(user_id, id))

    except NotImplementedError as e:
        return e
    except KeyError as e:
        return e
    except Exception as e:
        return e

@celery.task(name='retrain')
def retrain(user_id, id, train, y_train, test, y_test, params, cv):
    
    train = pd.DataFrame(json.loads(eval(train)))
    y_train = pd.Series(eval(y_train))
    if test is not None and y_test is not None:
        test = pd.DataFrame(json.loads(eval(test)))
        y_test = pd.Series(eval(y_test))
        
    try:
        return models_dao.retrain(user_id, id, train, y_train, test, y_test, params=params, cv=cv)
    except IndexError as e:
        return e
    except Exception as e:
        return e
            

@celery.task(name='delete_one')
def delete_one(user_id, id):
    try:
        models_dao.delete(user_id, id)
    except IndexError as e:
        return e
    except Exception as e:
        return e
    
@celery.task(name='train')
def train(user_id, name, train, y_train, test, y_test, params, cv):
    
    train = pd.DataFrame(json.loads(eval(train)))
    y_train = pd.Series(eval(y_train))
    if test is not None and y_test is not None:
        test = pd.DataFrame(json.loads(eval(test)))
        y_test = pd.Series(eval(y_test))
        
    try:
        return models_dao.train(user_id, name, train, y_train, test, y_test, params=params, cv=cv)
    except KeyError as e:
        return e
    except Exception as e:
        return e
    
@celery.task(name='predict')
def predict(user_id, id, test, y_test):
    
    try:
        test = pd.DataFrame(json.loads(eval(test)))
    except ValueError:
        test = pd.DataFrame(pd.Series(eval(test)))

    if test.shape[1] == 1:
        test = test.T
    if y_test is not None and type(y_test) != int:
        y_test = pd.Series(eval(y_test))
    
    try:
        return str(models_dao.predict(user_id, id, test, y_test))
    except IndexError as e:
        return e
    except KeyError as e:
        return e
    except Exception as e:
        return e


#####################################################################################

@celery.task(name='find_hyperparams')
def find_hyperparams(name, train, y_train, n_trials, cv, config):
    
    train = pd.DataFrame(json.loads(eval(train)))
    y_train = pd.Series(eval(y_train))
        
    return models_dao.find_hyperparams(name, train, y_train, n_trials=n_trials, cv=cv, conf=config)


