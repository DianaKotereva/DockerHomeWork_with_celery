import telebot
from telebot import types, apihelper
import requests
import numpy as np
import json
import pandas as pd
from pymongo import MongoClient
from config import TG_PROXY
import os

MONGO_URL = os.environ['MONGO_URL']
BOT_TOKEN = os.environ['BOT_TOKEN']

bot = telebot.TeleBot(BOT_TOKEN)
apihelper.proxy = {'http': TG_PROXY}
client = MongoClient(MONGO_URL)
db = client['testdb']
users = db['users']

class MongoDataBase:

    def __init__(self, MONGO_URL):

        client = MongoClient(MONGO_URL)
        self.users = client['testdb']['users']
        self.fields = ['id',
                       'ml_models', 'counter',
                       'num',
                       'train', 'y_train',
                       'test', 'y_test',
                       'config', 'find_params',
                       'cv', 'n_trials',
                       'pass_model', 'step',
                       'number', 'to_do']

        self.default_params = {'ml_models': [],
                               'counter': 0,
                               'params': {'num': 0,
                                          'train': None, 'y_train': None,
                                          'test': None, 'y_test': None,
                                          'config': str({}),
                                          'params': str({}),
                                          'find_params': None,
                                          'cv': 3,
                                          'n_trials': 20,
                                          'name': 'logreg',
                                          'step': '',
                                          'number': 0,
                                          'to_do': ''
                                          }
                               }

    def check_client(self, user_id):
        res = users.find_one({'id': user_id})
        if res is not None:
            return True
        else:
            self.add_client(user_id)
            return True

    def add_client(self, user_id):
        user_data = {'id': user_id}
        user_data.update(self.default_params)
        users.insert_one(user_data)
        return True

    def update_client(self, user_id, datas):
        users.update_one({'id': user_id}, {'$set': datas})

    def get_client(self, user_id):
        return users.find_one({'id': user_id})
    
    
mong = MongoDataBase(MONGO_URL)

class TelegramBot:
    def __init__(self):

        self.num = 0
        self.model_classes = {'logreg': 'Логистическая регрессия',
                              'forest': 'RandomForestClassifier', 'boosting': 'LGBMClassifier'}
        self.model_classes_inverse = {v: k for k, v in self.model_classes.items()}
        
        self.default_params = {'num': 0,
                               'train': None, 'y_train': None,
                               'test': None, 'y_test': None,
                               'config': str({}),
                               'params': str({}),
                               'find_params': None,
                               'cv': 3,
                               'n_trials': 20,
                               'name': 'logreg',
                               'step': '',
                               'number': 0,
                               'to_do': ''
                               }
        
        self.def_change = {'num': 0,
                               'config': str({}),
                               'params': str({}),
                               'find_params': None,
                               'cv': 3,
                               'n_trials': 20,
                               'name': 'logreg',
                               'step': '',
                               'number': 0,
                               'to_do': ''
                               }

    def get_params(self, user_id):
        return users.find_one({'id': user_id})['params']
    
    def update_params(self, user_id, params):
        users.update_one({'id': user_id}, {'$set': {'params': params}})
        
    def buttons(self, message, params):
        
        params.update(self.def_change)
        
        markup = types.ReplyKeyboardMarkup(
            one_time_keyboard=True, resize_keyboard=True)

        bt1 = types.KeyboardButton('Какие модели умеешь обучать?')
        bt2 = types.KeyboardButton('Обучить модель')
        bt3 = types.KeyboardButton('Какие модели обучены?')
        bt4 = types.KeyboardButton('Сколько моделей обучено?')
        bt5 = types.KeyboardButton('Получить модель')
        bt6 = types.KeyboardButton('Прогноз')
        bt7 = types.KeyboardButton('Delete')

        markup.add(bt1, bt2, bt3, bt4, bt5, bt6, bt7)

        if params['num'] == 0:
            question = 'Выберите, что вы хотите?'
        else:
            question = 'Что-то еще?'

        msg = bot.send_message(
            message.chat.id, text=question, reply_markup=markup)
        
        params['num'] += 1
        self.update_params(message.from_user.id, params)
        
        bot.register_next_step_handler(msg, self.make_res, params)

    def make_res(self, message, params):

        if message.text == 'Какие модели обучены?':
            self.get_models(message, params)
        elif message.text == 'Сколько моделей обучено?':
            self.get_count_models(message, params)
        elif message.text == 'Какие модели умеешь обучать?':
            bot.send_message(message.chat.id,
                             "Я умею обучать модели бинарной классификации: Логистическую регрессию, Случайный лес и Градиентный Бустинг")
            self.buttons(message, params)
            
        elif message.text == 'Получить модель':
            params['to_do'] = 'get'
            bot.send_message(message.chat.id, 'Нужен номер модели')
            self.select_one(message, params)
        elif message.text == 'Delete':
            params['to_do'] = 'delete'
            self.delete_choice(message, params)
        elif message.text == 'Обучить модель':
            params['to_do'] = 'train'
            self.train_choice(message, params)
        elif message.text == 'Прогноз':
            params['to_do'] = 'predict'
            bot.send_message(message.chat.id, 'Нужен номер модели')
            self.select_one(message, params)

    def train_choice(self, message, params):

        markup = types.ReplyKeyboardMarkup(
            one_time_keyboard=True, resize_keyboard=True)
        bt1 = types.KeyboardButton('Новая модель')
        bt2 = types.KeyboardButton('Переобучить модель')
        markup.add(bt1, bt2)

        question = 'Что вы хотите обучить?'
        msg = bot.send_message(
            message.chat.id, text=question, reply_markup=markup)
        bot.register_next_step_handler(msg, self.train_sel, params)

    def train_sel(self, message, params):

        if message.text == 'Новая модель':

            params['to_do'] = 'train_new'

            markup = types.ReplyKeyboardMarkup(
                one_time_keyboard=True, resize_keyboard=True)
            bt1 = types.KeyboardButton('Логистическая регрессия')
            bt2 = types.KeyboardButton('RandomForestClassifier')
            bt3 = types.KeyboardButton('LGBMClassifier')
            markup.add(bt1, bt2, bt3)

            question = 'Что вы хотите обучить?'
            msg = bot.send_message(
                message.chat.id, text=question, reply_markup=markup)
            bot.register_next_step_handler(message, self.select_type, params)

        elif message.text == 'Переобучить модель':

            params['to_do'] = 'retrain'
            bot.send_message(
                message.chat.id, 'Какую модель? Нужен номер изменяемой модели')
            self.select_one(message, params)

    def select_type(self, message, params):
    
        params['step'] = 'Обучить модель'
        
        if params['to_do'] == 'train_new':
            params['name'] = self.model_classes_inverse[message.text]

        bot.send_message(message.chat.id, 'Проверим train данные')
        if params['train'] is None or params['y_train'] is None:
            bot.send_message(
                message.chat.id, 'Train данных нет. Пожалуйста, пришлите train датасет (без таргета) в формате json:')
            bot.register_next_step_handler(message, self.get_train_data, params)
        else:
            self.yes_no_choice(message, params)

    def yes_no_choice(self, message, params):

        markup = types.ReplyKeyboardMarkup(
            one_time_keyboard=True, resize_keyboard=True)
        bt1 = types.KeyboardButton('Да')
        bt2 = types.KeyboardButton('Нет')
        markup.add(bt1, bt2)
        
        if params['step'] == 'Обучить модель':
            question = 'Сейчас в памяти имеются train данные. Хотите поменять?'
        elif params['step'] == 'Подобрать гиперпараметры':
            question = 'Вы хотите подобрать гиперпараметры с помощью optuna?'
        elif params['step'] == 'Дополнительные параметры подбора':
            question = 'Вы хотите задать cv и n_trials (для optuna). Default: cv = 3, n_trials = 20? Если датасет маленький, лучше поставить cv=1'
        elif params['step'] == 'Задать гиперпараметры':
            question = 'Вы хотите задать свои гиперпараметры для подбора/обучения?'
        elif params['step'] == 'Получить тестовые данные':
            question = 'Вы хотите использовать y_test для оценки?'

        msg = bot.send_message(
            message.chat.id, text=question, reply_markup=markup)
        bot.register_next_step_handler(msg, self.check_yes_no, params)

    def check_yes_no(self, message, params):

        if params['step'] == 'Обучить модель':
            if message.text == 'Да':
                bot.send_message(
                    message.chat.id, 'Хорошо. Пожалуйста, пришлите train датасет (без таргета) в формате json')
                bot.register_next_step_handler(message, self.get_train_data, params)
            elif message.text == 'Нет':
                params['step'] = 'Подобрать гиперпараметры'
                self.yes_no_choice(message, params)

        elif params['step'] == 'Подобрать гиперпараметры':
            if message.text == 'Да':
                params['find_params'] = True
                params['step'] = 'Дополнительные параметры подбора'
                self.yes_no_choice(message, params)
            elif message.text == 'Нет':
                params['find_params'] = None
                params['step'] = 'Дополнительные параметры подбора'
                self.yes_no_choice(message, params)

        elif params['step'] == 'Дополнительные параметры подбора':
            if message.text == 'Да':
                bot.send_message(message.chat.id, 'Хорошо. Введите cv')
                bot.register_next_step_handler(message, self.get_cv, params)
            elif message.text == 'Нет':
                params['step'] = 'Задать гиперпараметры'
                self.yes_no_choice(message, params)

        elif params['step'] == 'Задать гиперпараметры':
            if message.text == 'Да':
                bot.send_message(
                    message.chat.id, 'Хорошо. Пожалуйста, введите строку с config')
                bot.register_next_step_handler(message, self.get_config, params)
            else:
                bot.send_message(
                    message.chat.id, 'Хорошо. Начинаю обучать модель')
                self.train_model(message, params)

        elif params['step'] == 'Получить тестовые данные':
            if message.text == 'Да':
                bot.send_message(
                    message.chat.id, 'Хорошо. Пожалуйста, пришлите y_test array в формате json')
                bot.register_next_step_handler(message, self.get_y_test_data, params)
            else:
                params['y_test'] = None
                bot.send_message(message.chat.id, 'Хорошо. Начинаю прогноз')
                self.predict(message, params)

    def get_cv(self, message, params):

        try:
            params['cv'] = int(message.text)
            bot.send_message(message.chat.id, 'Хорошо. Введите n_trials')
            bot.register_next_step_handler(message, self.get_n_trials, params)
        except ValueError:
            bot.send_message(message.chat.id, 'Цифрами, пожалуйста')
            bot.register_next_step_handler(message, self.get_cv, params)

    def get_n_trials(self, message, params):

        try:
            params['n_trials'] = int(message.text)
            params['step'] = 'Задать гиперпараметры'
            self.yes_no_choice(message, params)
        except ValueError:
            bot.send_message(message.chat.id, 'Цифрами, пожалуйста')
            bot.register_next_step_handler(message, self.get_n_trials, params)

    def get_config(self, message, params):

        params['config'] = message.text
        params['params'] = message.text
        bot.send_message(message.chat.id, 'Хорошо. Начинаю обучать модель')
        self.train_model(message, params)

    def get_train_data(self, message, params):

        try:
            if message.content_type == 'document':
                file_info = bot.get_file(message.document.file_id)
                params['train'] = bot.download_file(file_info.file_path)
                params['train'] = json.dumps(params['train'].decode('utf-8'))
            elif message.content_type == 'text':
                params['train'] = message.text
#             params['train'] = pd.DataFrame(json.loads(params['train']))

        except Exception as e:
            bot.reply_to(message, e)

        bot.send_message(
            message.chat.id, 'Хорошо. Пожалуйста, пришлите y_train array в формате json')
        
        self.update_params(message.from_user.id, params)
        bot.register_next_step_handler(message, self.get_y_train_data, params)

    def get_y_train_data(self, message, params):

        try:
            if message.content_type == 'document':
                file_info = bot.get_file(message.document.file_id)
                params['y_train'] = bot.download_file(file_info.file_path)
                params['y_train'] = params['y_train'].decode('utf-8')

            elif message.content_type == 'text':
                params['y_train'] = message.text

            params['y_train'] = eval(params['y_train'])
            if (len(params['y_train']) == 1 and type(params['y_train'][list(params['y_train'].keys())[0]]) == dict):
                params['y_train'] = y_train[list(params['y_train'].keys())[0]]
            params['y_train'] = pd.Series(params['y_train'])

            if len(np.unique(params['y_train'])) > 2:
                bot.send_message(
                    message.chat.id, 'Класс обучает БИНАРНЫЕ классификации! Пришлите таргет еще раз')
                bot.register_next_step_handler(message, self.get_y_train_data, params)
            else:
                params['step'] = 'Подобрать гиперпараметры'
                params['y_train'] = params['y_train'].to_json()
                self.yes_no_choice(message, params)

        except Exception as e:
            bot.reply_to(message, e)

    def get_test_data(self, message, params):

        try:
            if message.content_type == 'document':
                chat_id = message.chat.id
                file_info = bot.get_file(message.document.file_id)
                params['test'] = bot.download_file(file_info.file_path)
                params['test'] = json.dumps(params['test'].decode('utf-8'))
            elif message.content_type == 'text':
                params['test'] = message.text

#             params['test'] = pd.DataFrame(json.loads(params['test']))
        except Exception as e:
            bot.reply_to(message, e)

        params['step'] = 'Получить тестовые данные'
        self.yes_no_choice(message, params)

    def get_y_test_data(self, message, params):

        try:
            if message.content_type == 'document':
                chat_id = message.chat.id
                file_info = bot.get_file(message.document.file_id)
                params['y_test'] = bot.download_file(file_info.file_path)
                params['y_test'] = params['y_test'].decode('utf-8')
            elif message.content_type == 'text':
                params['y_test'] = message.text
            params['y_test'] = eval(params['y_test'])
            if (len(params['y_test']) == 1 and type(params['y_test'][list(params['y_test'].keys())[0]]) == dict):
                params['y_test'] = params['y_test'][list(params['y_test'].keys())[0]]
            params['y_test'] = pd.Series(params['y_test'])

            if len(np.unique(params['y_test'])) > 2:
                bot.send_message(
                    message.chat.id, 'Класс обучает БИНАРНЫЕ классификации! Пришлите таргет еще раз')
                bot.register_next_step_handler(message, self.get_y_test_data, params)
            else:
                bot.send_message(message.chat.id, 'Начинаю прогноз')
                params['y_test'] = params['y_test'].to_json()
                self.predict(message, params)
        except Exception as e:
            bot.reply_to(message, e)

    def predict(self, message, params):

        try:
            url = "http://host.docker.internal:5000/api/ml_models/predict/" + \
                str(params['number'])
            if params['y_test'] is not None:
                resp = requests.get(
                    url, data={'test': params['test'], 'y_test': params['y_test'], 'user_id': message.from_user.id})
            else:
                resp = requests.get(url, data={'test': params['test'], 'user_id': message.from_user.id})
            res = eval(eval(resp.text))
            if type(res) == list:
                # Отправить в виде json/txt файла
                with open('predict.txt', 'w') as file:
                    file.write(str(res))
                f = open("predict.txt", "rb")
                bot.send_document(message.chat.id, f)

            elif type(res) == dict:
                # Отправить predict_proba в виде json/txt файла
                # Напечатать результат
                if 'message' in list(res.keys()):
                    bot.send_message(
                        message.chat.id, f"Ошибка: {res['message']}")
                else:
                    with open('predict.txt', 'w') as file:
                        file.write(str(res['predict_proba']))
                    f = open("predict.txt", "rb")
                    bot.send_document(message.chat.id, f)
                    bot.send_message(
                        message.chat.id, f"Модель {params['number']}. \n Accuracy: {res['acc']}. \n ROC_AUC: {res['roc_auc']}. \n APS: {res['aps']}")

            self.buttons(message, params)
        except Exception as e:
            bot.send_message(message.chat.id, f"Error: {e}")
            self.buttons(message, params)

    def train_model(self, message, params):

        datas = params
        datas.update({'user_id': message.from_user.id})
        
        type(datas['train'])
#         print(datas)

        try:
            if params['to_do'] == 'train_new':
                resp = requests.post(
                    "http://host.docker.internal:5000/api/ml_models/train", data=datas)
                res = eval(resp.text)
            elif params['to_do'] == 'retrain':
                url = "http://host.docker.internal:5000/api/ml_models/"+str(params['number'])
                resp = requests.put(url, data=datas)
                res = eval(resp.text)

            if 'message' in list(res.keys()):
                bot.send_message(
                    message.chat.id, f"Модель не обучена, ошибка: {res['message']}")

                params['train'] = None
                params['y_train'] = None

            else:
                bot.send_message(
                    message.chat.id, f"Модель обучена\n Accuracy: {res['acc']}. \n ROC_AUC: {res['roc_auc']}. \n APS: {res['aps']}")
            self.buttons(message, params)
        except Exception as e:
            bot.send_message(message.chat.id, f"Error: {e}")
            self.buttons(message, params)

    def get_models(self, message, params):

        resp = requests.get("http://host.docker.internal:5000/api/ml_models", data = {'user_id': message.from_user.id})
        res = eval(eval(resp.text))
        if len(res) > 0:
            for num in range(len(res)):
                r = res[num]
                bot.send_message(message.chat.id,
                                 f"Модель {num}. Тип: {self.model_classes[r['name']]}. \n Accuracy: {r['acc']}. \n ROC_AUC: {r['roc_auc']}. \n APS: {r['aps']}")
        else:
            bot.send_message(message.chat.id,
                             "Еще не обучено ни одной модели. Самое время начать!")
        self.buttons(message, params)

    def get_count_models(self, message, params):

        resp = requests.get("http://host.docker.internal:5000/api/ml_models/count", data = {'user_id': message.from_user.id})
        res = eval(eval(resp.text))
        bot.send_message(message.chat.id, res)
        self.buttons(message, params)

    def delete_choice(self, message, params):

        markup = types.ReplyKeyboardMarkup(
            one_time_keyboard=True, resize_keyboard=True)
        bt1 = types.KeyboardButton('Все')
        bt2 = types.KeyboardButton('Одну модель')
        markup.add(bt1, bt2)

        question = 'Что вы хотите удалить?'
        msg = bot.send_message(
            message.chat.id, text=question, reply_markup=markup)
        bot.register_next_step_handler(msg, self.delete, params)

    def delete(self, message, params):

        if message.text == 'Все':
            resp = requests.delete("http://host.docker.internal:5000/api/ml_models", data = {'user_id': message.from_user.id})
            bot.send_message(
                message.chat.id, 'Все модели удалены, возвращаюсь на стартовое меню')
            self.buttons(message, params)

        elif message.text == 'Одну модель':
            bot.send_message(
                message.chat.id, 'Какую модель? Нужен номер удаляемой модели')
            self.select_one(message, params)

    def select_one(self, message, params):

        markup = types.ReplyKeyboardMarkup(
            one_time_keyboard=True, resize_keyboard=True)
        resp = requests.get("http://host.docker.internal:5000/api/ml_models/count", data = {'user_id': message.from_user.id})
        res = eval(eval(resp.text))
        if res == 0:
            bot.send_message(
                message.chat.id, 'Простите, но пока не обучено ни одной модели.')
            self.buttons(message, params)
        else:
            btts = []
            for n in range(res):
                bt1 = types.KeyboardButton(str(n))
                btts.append(bt1)

            markup.add(*btts)

            question = 'Выберите модель'
            msg = bot.send_message(
                message.chat.id, text=question, reply_markup=markup)
            bot.register_next_step_handler(msg, self.process_one, params)

    def process_one(self, message, params):

        params['number'] = int(message.text)
        number = params['number']
        url = "http://host.docker.internal:5000/api/ml_models/"+str(number)
        if params['to_do'] == 'delete':
            resp = requests.delete(url, data = {'user_id': message.from_user.id})
            bot.send_message(message.chat.id, f'Модель {number} удалена')
            self.buttons(message, params)

        elif params['to_do'] == 'retrain':
            self.select_type(message, params)

        elif params['to_do'] == 'get':
            resp = requests.get(url, data = {'user_id': message.from_user.id})
            r = eval(eval(resp.text))
            bot.send_message(
                message.chat.id, f"Модель {number}. Тип: {self.model_classes[r['name']]}. \n Accuracy: {r['acc']}. \n ROC_AUC: {r['roc_auc']}. \n APS: {r['aps']}")
            self.buttons(message, params)

        elif params['to_do'] == 'predict':
            bot.send_message(
                message.chat.id, 'Загрузите test данные в формате json')
            bot.register_next_step_handler(message, self.get_test_data, params)
            
            
telegram_bot = TelegramBot()

@bot.message_handler(content_types=['text', 'document'])
def start(message):
    bot.send_message(message.from_user.id, 
                         "Привет, я ML бот! Я умею обучать ML модели. Введите /start, поучим модели!")
    if message.text == '/start':
        mong.check_client(message.from_user.id)
        params = telegram_bot.get_params(message.from_user.id)
        params['num'] = 0
        bot.register_next_step_handler(message, telegram_bot.buttons, params) 
    else:
        bot.send_message(message.from_user.id, 'Введите /start');
        
bot.polling(none_stop=True, interval=0)
    
    
    
    