# DockerHomeWork_with_celery
 
Для запуска проекта:

Клонировать проект, перейти в папку проекта.
Проект использует подключение к telegram и mongo, поэтому для работы требуется создать .env с BOT_TOKEN и строкой подключения к mongo и заменить путь к env-file в docker-compose

Далее запустить в терминале следующие команды:
docker-compose build
docker-compose up

Протестировать решение можно через телеграм бот или с помощью ipynb файла в проекте. 

Работа с БД:

В качестве БД использовался docker образ mongo.
В БД хранятся этапы взаимодействия пользователя (по user_id в телеграмме) с ботом, а также счетчик количества моделей и сами модели. 

Вычисления не в рамках Flask-приложения:
Все этапы обучения моделей и обновление БД происходят через celery не в рамках Flask приложения.

Образы в docker-hub:

https://hub.docker.com/repository/docker/1306613066/telegram_image
https://hub.docker.com/repository/docker/1306613066/worker_image
https://hub.docker.com/repository/docker/1306613066/app_image

