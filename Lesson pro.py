# Перед запуском требуется:
# source myenv/bin/activate
# pip install torch
# pip install ultralytics
# pip install python-telegram-bot

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
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
async def saving(update, context):
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
    # сохраняем имя изображения в user_data
    context.user_data['image_name'] = image_name
    # скачиваем файл с сервера Telegram в папку images
    await new_file.download_to_drive(image_path)

    # создаем список Inline кнопок
    keyboard = [[InlineKeyboardButton("Люди", callback_data="0"),
                InlineKeyboardButton("Авто", callback_data="2")]]
    
    # создаем Inline клавиатуру
    reply_markup = InlineKeyboardMarkup(keyboard)

    # прикрепляем клавиатуру к сообщению
    await update.message.reply_text('Какой класс объектов распознавать?', reply_markup=reply_markup)


async def detection(update, context):
    i = 0
    exp_num = 1
    # параметры входящего запроса при нажатии на кнопку
    query = update.callback_query

    # Получаем имя изображения из контекста
    image_name = context.user_data.get('image_name')

    if not image_name:
        await update.callback_query.message.reply_text('Ошибка: изображение не найдено')
        return

    # Запускаем цикл распознаваний
    for conf in conf_list:
        for iou in iou_list:
            # создаем словарь с параметрами
            test_dict = dict()
            test_dict['weights'] = 'yolov5x.pt'     # Самые сильные веса yolov5x.pt, вы также можете загрузить версии: yolov5n.pt, yolov5s.pt, yolov5m.pt, yolov5l.pt (в порядке возрастания)

            test_dict['source'] = 'images'          # папка, в которую загружаются присланные в бота изображения
            test_dict['conf'] = conf                # порог распознавания
            test_dict['classes'] = query.data        # классы, которые будут распознаны       
            test_dict['iou-thres'] = iou  # Порог IoU

            # вызов функции detect из класса TerraYolo)
            yolov5.run(test_dict, exp_type='test')
               
            # отправляем пользователю результат
            await update.callback_query.message.reply_text(f'Распознавание {"людей" if query.data == "0" else "авто"} с достоверностью {conf} и пересечением {iou} завершено') # отправляем пользователю результат 
            await update.callback_query.message.reply_photo(f"{WORK_DIR}/yolov5/runs/detect/exp{exp_num if exp_num > 1 else ''}/{image_name}") # отправляем пользователю результат изображение
            exp_num += 1

    
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
    application.add_handler(MessageHandler(filters.PHOTO, saving, block=False))
    application.add_handler(MessageHandler(filters.TEXT, help))
    # добавляем обработчик нажатия Inline кнопок
    application.add_handler(CallbackQueryHandler(detection))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document, block=False))
   
    application.run_polling() # запускаем бота (остановка CTRL + C)


if __name__ == "__main__":
    main()
