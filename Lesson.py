# Перед запуском требуется:
# source myenv/bin/activate
# pip install torch
# pip install ultralytics
# pip install python-telegram-bot

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
import os
import shutil
from TerraYolo.TerraYolo import TerraYoloV5             # загружаем фреймворк TerraYolo
from PIL import Image
# import torch

# возьмем переменные окружения из .env
load_dotenv()

# загружаем токен бота
TOKEN =  os.environ.get("TOKEN") # ВАЖНО !!!!!
  

# инициализируем класс YOLO
WORK_DIR = r'/home/andrey/Bots/Yolo_Bot'
os.makedirs(WORK_DIR, exist_ok=True)
yolov5 = TerraYoloV5(work_dir=WORK_DIR)


# функция команды /start
async def start(update, context):
    await update.message.reply_text('Пришлите фото для распознавания объектов')

# функция для работы с текстом
async def help(update, context):
    await update.message.reply_text(update)

# Задаём значения для переборов
conf_list = [0.01, 0.5, 0.99]
iou_list = [0.01, 0.5, 0.99]

# функция обработки изображения
async def detection(update, context):
    # удаляем папку images с предыдущим загруженным изображением и папку runs с результатом предыдущего распознавания
    try:
        shutil.rmtree('images') 
        shutil.rmtree(f'{WORK_DIR}/yolov5/runs') 
    except:
        pass

    my_message = await update.message.reply_text('Мы получили от тебя фотографию. Идет распознавание объектов...')
    # получение файла из сообщения
    new_file = await update.message.photo[-1].get_file()

    # имя файла на сервере
    os.makedirs('images', exist_ok=True)
    image_name = str(new_file['file_path']).split("/")[-1]
    image_path = os.path.join('images', image_name)
    # скачиваем файл с сервера Telegram в папку images
    await new_file.download_to_drive(image_path)

    i = 0
    exp_num = 1

    # Запускаем цикл распознаваний
    for conf in conf_list:
        for iou in iou_list:
            # создаем словарь с параметрами
            test_dict = dict()
            test_dict['weights'] = 'yolov5x.pt'     # Самые сильные веса yolov5x.pt, вы также можете загрузить версии: yolov5n.pt, yolov5s.pt, yolov5m.pt, yolov5l.pt (в порядке возрастания)

            test_dict['source'] = 'images'          # папка, в которую загружаются присланные в бота изображения
            test_dict['conf'] = conf                # порог распознавания
            test_dict['classes'] = '0 2'        # классы, которые будут распознаны
            # 'img-size': 640,  # Размер изображения
            # 'conf-thres': 0.25,  # Порог уверенности
            test_dict['iou-thres'] = iou  # Порог IoU

            # вызов функции detect из класса TerraYolo)
            yolov5.run(test_dict, exp_type='test')

            
            # удаляем предыдущее сообщение от бота
            # print('my_message = ', my_message)
            # await update.message.reply_text(f'my_message = {my_message}')
            if i < 1:
                await context.bot.deleteMessage(message_id = my_message.message_id, # если не указать message_id, то удаляется последнее сообщение
                                                chat_id = update.message.chat_id) # если не указать chat_id, то удаляется последнее сообщение
                i += 1

            # отправляем пользователю результат
            await update.message.reply_text(f'Распознавание объектов с достоверностью {conf} и пересечением {iou} завершено') # отправляем пользователю результат 
            await update.message.reply_photo(f"{WORK_DIR}/yolov5/runs/detect/exp{exp_num if exp_num > 1 else ''}/{image_name}") # отправляем пользователю результат изображение
            exp_num += 1


async def detection_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Проверка формата и размера изображения
    # Получаем документ
    document = update.message.document
    await update.message.reply_text(f'Получен не сжатый документ')
    # Проверяем MIME-тип
    if document.mime_type.startswith('image/'):
        # detection(update, context)
        await update.message.reply_text(f'Вы отправили изображение: {document.file_name}')
        # удаляем папку images с предыдущим загруженным изображением и папку runs с результатом предыдущего распознавания
        try:
            shutil.rmtree('images') 
            shutil.rmtree(f'{WORK_DIR}/yolov5/runs') 
        except:
            pass

        # получение файла из сообщения
        new_file = await document.get_file()

        # имя файла на сервере
        os.makedirs('images', exist_ok=True)
        image_name = str(new_file['file_path']).split("/")[-1]
        image_path = os.path.join('images', image_name)
        # скачиваем файл с сервера Telegram в папку images
        await new_file.download_to_drive(image_path)

        i = 0
        exp_num = 1

        # Запускаем цикл распознаваний
        for conf in conf_list:
            for iou in iou_list:
                # создаем словарь с параметрами
                test_dict = dict()
                test_dict['weights'] = 'yolov5x.pt'     # Самые сильные веса yolov5x.pt, вы также можете загрузить версии: yolov5n.pt, yolov5s.pt, yolov5m.pt, yolov5l.pt (в порядке возрастания)

                test_dict['source'] = 'images'          # папка, в которую загружаются присланные в бота изображения
                test_dict['conf'] = conf                # порог распознавания
                test_dict['classes'] = '0 2'        # классы, которые будут распознаны
                # 'img-size': 640,  # Размер изображения
                # 'conf-thres': 0.25,  # Порог уверенности
                test_dict['iou-thres'] = iou  # Порог IoU

                # вызов функции detect из класса TerraYolo)
                yolov5.run(test_dict, exp_type='test')

            
                # # удаляем предыдущее сообщение от бота
                # # print('my_message = ', my_message)
                # # await update.message.reply_text(f'my_message = {my_message}')
                # if i < 1:
                #     await context.bot.deleteMessage(message_id = my_message.message_id, # если не указать message_id, то удаляется последнее сообщение
                #                                 chat_id = update.message.chat_id) # если не указать chat_id, то удаляется последнее сообщение
                #     i += 1

                # отправляем пользователю результат
                await update.message.reply_text(f'Распознавание объектов с достоверностью {conf} и пересечением {iou} завершено') # отправляем пользователю результат 
                await update.message.reply_photo(f"{WORK_DIR}/yolov5/runs/detect/exp{exp_num if exp_num > 1 else ''}/{image_name}") # отправляем пользователю результат изображение
                exp_num += 1


    else:
        await update.message.reply_text(f'Вы отправили документ: {document.file_name}, но это не изображение.')
    #     # Здесь вы можете обработать изображение
    #     file_id = document.file_id
    #     file_name = document.file_name
    #     await context.bot.get_file(file_id).download_to_drive(file_name)
    #     # Добавьте вашу логику обработки изображения здесь
    # else:
    #     # Обработка других типов документов, если необходимо
    #     pass
    # ############
    #         # Также можно проверить формат
    #         if img.format not in ['JPEG', 'PNG']:
    #             await update.message.reply_text('Пожалуйста, отправьте изображение в формате JPEG или PNG.')
    #             return
    
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
       document = update.message.document
       file = await document.get_file()
       
       # Сохраните файл
       await file.download_to_drive(f"{document.file_name}")

       await update.message.reply_text("Документ успешно получен и сохранен!")


def main():

    # точка входа в приложение
    application = Application.builder().token(TOKEN).build() # создаем объект класса Application
    print('Бот запущен...')

    # добавляем обработчик команды /start
    application.add_handler(CommandHandler("start", start))
    # добавляем обработчик изображений, которые загружаются в Telegram в СЖАТОМ формате
    # (выбирается при попытке прикрепления изображения к сообщению)
    application.add_handler(MessageHandler(filters.PHOTO, detection, block=False))
    application.add_handler(MessageHandler(filters.TEXT, help))

    # добавляем обработчика для НЕ СЖАТЫХ фото
    # application.add_handler(MessageHandler(filters.Document, detection_document, block=False))
    application.add_handler(MessageHandler(filters.Document.ALL, detection_document, block=False))  
    # application.add_handler(MessageHandler(filters.Document | filters._Photo, detection, block=False))

    application.run_polling() # запускаем бота (остановка CTRL + C)


if __name__ == "__main__":
    main()
