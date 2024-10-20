import os
import random
import shutil
from pathlib import Path

import yaml

from .dataset_utils import count_classes_in_lab_dir, show_list_images, \
    show_class_images
import sys
if 'google.colab' in sys.modules:
    from .colab_os_com import colab_os_com as os_com
else:
    from .non_colab_os_com import non_colab_os_com as os_com


# print(f'{__file__}, {__name__}, {__package__}')

def get_last_exp_dir(proj_path, name='exp'):
    """
    :param proj_path: путь для поиска - соответствует пути для параметра --project
    :param name: соответствует параметру --name
    :return: путь к каталогу с последним экспериментом
    NOTE: установка и сброс параметра --exist-ok, сбои и прерывания работы программы,
    редактирование содержимого диска могут приводить к неправильному результату
    следующая версия этой функции должна искать веса и предлагать последнюю по времени
    сохранения папку weight

    # ниже как имя подкаталога задается в yolo
    # ищем по имени каталога (по умолчанию - 'exp') каталог с максимальным индексом
    #for n in range(2, 9999):
        #p = f'{path}{sep}{n}{suffix}'  # increment path
        #if not os.path.exists(p):  #
            #break
    #path = Path(p)

    """


    last_exp_dir = ''
    if os.path.isdir(proj_path):
        last_exp_dir = proj_path + os.sep + name
        if os.path.isdir(last_exp_dir):
            # был как минимум один эксперимент
            exp_dir_list = sorted([
                exp_dir for exp_dir in os.listdir(proj_path) if
                (os.path.isdir(proj_path + os.sep + exp_dir)) &
                (exp_dir.find(name) == 0)
            ])
            exp_dir_list = exp_dir_list[1:]  # удалили каталог '/exp/'
            if len(exp_dir_list) >= 1:
                exp_dir_num = max(
                    [int(exp_dir[len(name):]) for exp_dir in exp_dir_list])
                # print('exp_dir_num', exp_dir_num)
                last_exp_dir = last_exp_dir + str(exp_dir_num)

    return last_exp_dir


def get_model_last_exp_dir(model, sub='train'):
    """
    ищет каталог с последним экспериментом (тип )
    :param model: объект типа TerraYoloV5
    :param sub:  тип эксперимента 'train', 'val', 'test'
    :return: возвращает найденный путь или None
    Note при сложной схеме экспериментов, например, при использовании параметра
    "exist-ok" можно получить ошибочный результат
    """
    #
    exp_dict = model.exp_dict[sub]
    try:
        # в приоритете текущая история модели
        last_best_weights_path = exp_dict['last_exp_path']
    except:
        proj_path = model.model_dir + os.sep + exp_dict.get('def_dir')
        # здесь можно попытаться извлечь name из exp_dict['yolo_param_dict'] но
        # пока не представляю ситуации, что last_exp_path нет,
        # а exp_dict['yolo_param_dict']['name'] - есть
        last_best_weights_path = get_last_exp_dir(proj_path=proj_path, name='exp')
    return last_best_weights_path


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "(" + str(counter) + ")" + extension
        counter += 1

    return path


def save_data_yaml(yaml_dict, dest_yaml_path, f_overwrite=False,
                   key_set=frozenset(['train', 'val', 'nc', 'names'])):
    """
    Заполняет описание датасета по шаблону и сохраняет по указанному пути.

    :param yaml_dict: параметры описания
    :param dest_yaml_path: путь к описанию
    :return:
    """
    '''
    yaml_content = f"""
        train: {yaml_dict['train']}
        val: {yaml_dict['val']}
        test: {yaml_dict.get('test', '')}

        # number of classes
        nc: {yaml_dict['nc']}

        # class names
        names: {yaml_dict['names']}
        """
    '''
    keys_missing = key_set - set(yaml_dict.keys())
    if keys_missing:
        print(f'Отсутствуют необходимые ключи словаря {keys_missing}')

    if os.path.exists(dest_yaml_path):
        print(f'файл {dest_yaml_path} существует')
        if not f_overwrite:
            dest_yaml_path = uniquify(dest_yaml_path)

    with open(dest_yaml_path, 'w') as f:
        #f.write(yaml_content)
        yaml.dump(yaml_dict, f, default_flow_style=None)
    print(f'Файл {dest_yaml_path} сохранен')

    return dest_yaml_path



def get_data_dict(yaml_path_or_dict):
    """
    Загружает в словарь файл типа yaml. Применяется для описаний датасетов
    :param yaml_path:
    :return:
    """
    if isinstance(yaml_path_or_dict, (dict)):
        return yaml_path_or_dict
    try:
        with open(yaml_path_or_dict, 'r') as f:
            return yaml.safe_load(f)
    except:
        return {}

def get_lab_img_dir(img_lab_dir):
    """
    Для каталога аканчивающегося на images заменит images на labels
    иначе - заменит ПОСЛЕДНИЙ элемент на images
    :param img_lab_dir: каталог images или  labels
    :return:
    NOTE если не images - проверка на labels не делается
    """
    dir_list = img_lab_dir.split(os.sep)
    if dir_list[-1]=='images':
        return os.sep.join(dir_list[:-1]+['labels'])
    else:
        return os.sep.join(dir_list[:-1]+['images'])

def get_actual_yaml(yaml_path):
    pass


class TerraYoloV5:

    """
    Класс загружает yolov5 и запускает train.py, val.py, detect.py
    скрипты конфинурируются при помощи словарей
    exp_dict['train'], exp_dict['test'], exp_dict['val']

    train_dict=dict()
    train_dict['data'] = data_path #путь к описанию датасета
    train_dict['epochs'] = 50
    #train_dict['img'] = 320

    my_terra_yolov5.run(train_dict, exp_type='train')

    test_dict=dict()
    test_dict['weights'] = os.path.join(
            my_terra_yolov5.exp_dict['train']['last_exp_path'], 'weights', 'best.pt')
    test_dict['source'] = ''#путь
    test_dict['save-txt'] = ''
    my_terra_yolov5.run(test_dict,'test')
    """

    def __init__(self,
                 # data_path: str, #путь к файлу описания датасета.yaml
                 work_dir='.',
                 model_ref='https://github.com/ultralytics/yolov5',
                 log_f_name='model.log',
                 reload=False,
                 backup_dir=None,
                 ):
        """
        Создаем рабочий каталог и загружаем в него модель

        :param work_dir: рабочий каталог
        :param model_ref: ссылка на git
        :param log_f_name: лог - файл, используется для отладки
        :param reload: если True полностью удаляем каталог с моделью и загружаем снова
            если  False перезагружаем только в случае отсутствия каталога с моделью

        """

        if not os.path.exists(work_dir):
            os.makedirs(work_dir, exist_ok=True)
        self.work_dir = os.path.abspath(work_dir + os.sep)

        self.log_f_name = work_dir + os.sep + log_f_name
        self.model_ref = model_ref
        # self.model_name = model_name
        self.reload = reload
        self.model_dir = self.work_dir + os.sep + 'yolov5/'
        self.download()
        self.model_dict = self.get_model_dict()

        # exp_dict = словарь с результатами экспериментов
        self.exp_dict = dict()
        for sub in ['train', 'val', 'test']:
            self.exp_dict[sub] = dict()

        self.exp_dict['train']['def_dir'] = '/runs/train/'
        self.exp_dict['val']['def_dir'] = '/runs/val/'
        self.exp_dict['test']['def_dir'] = '/runs/detect/'

        self.exp_dict['train']['script'] = 'train.py'
        self.exp_dict['val']['script'] = 'val.py'
        self.exp_dict['test']['script'] = 'detect.py'
        self.exp_dict['detect'] = self.exp_dict['test']

        if not backup_dir:
            self.backup_dir = os.path.join(self.work_dir, 'backup')

    def download(self):

        """
        метод в зависимости от значения параметров self.reload
        проверяет наличие каталога self.work_dir и загружает в него модель
        если self.reload == True полностью удаляем каталог с моделью и загружаем снова
        если  self.reload == False перезагружаем только в случае отсутствия каталога
        с моделью
        :return:
        """

        if os.path.isdir(self.model_dir) == True:
            if self.reload == True:
                shutil.rmtree(self.model_dir)
            else:
                # os.chdir(self.model_dir)
                # self.os_com(f'pip install -r requirements.txt')
                # os.chdir(wd)
                return

        wd = os.getcwd()
        os.chdir(self.work_dir)

        self.os_com(f'git clone {self.model_ref}')
        os.chdir(self.model_dir)
        self.os_com('pip install -r requirements.txt')
        os.chdir(wd)
        return

    def get_model_dict(self):

        """

        :return:
        метод возвращает словарь имя файла: путь к найденными файлам описаний моделей
        NOTE: в первом приближении пропустил модели .py
        """

        model_path = Path(self.model_dir + '/models/')
        model_full_path = sorted(list(model_path.rglob('*.yaml')))
        model_names = [str(path)[len(str(model_path)) + 1:].split('.')[0]
                       for path in model_full_path]
        # return model_short_path
        return dict(zip(model_names, model_full_path))

    def get_annotation(self, model_name):

        """
        метод выводит на экран содержимое файла описнаия по имени модели
        :param model_name:
        :return:
        NOTE: пока yaml, словарь если что сделаем
        """

        with open(self.model_dict[model_name], 'r') as f:
            print(f.read())

    def do_experiment(self, command, param_dict, project_dir):

        """

        :param command:
        :param param_dict:
        :param project_dir:
        :return:

        NOTE:
         чуть позже написал для определения каталога с результатами функции
         get_model_last_exp_dir, get_last_exp_dir
         но все равно история экспериментов может влисять на результат
         возможны запуск и работа с одной моделью, прописывание путей в параметрах
         'exist-ok', неудачное завершение экспериментов, потому
         в некоторых сложных случаях последние веса придется искать экспериментатору

        """

        exp_dir_set = set()
        if os.path.isdir(project_dir) == True:
            exp_dir_set = set([dir_name for dir_name in os.listdir(project_dir)
                               if os.path.isdir(project_dir + os.sep + dir_name)])
        # print('exp_dir_set', exp_dir_set)

        self.os_com(command, log_f=False)

        last_train_subdir = ''
        if os.path.isdir(project_dir) == True:

            if 'exist-ok' in param_dict.keys():
                # если параметр 'exist-ok' задан
                # результат эксперимента сохранится в подкаталоге param_dict['name']
                # (по-умолчанию param_dict['name']='exp')
                last_train_subdir = param_dict.get('name', 'exp')
            else:
                # если параметр 'exist-ok' не задан - должен быть один новый каталог
                # но если что-то пойдет не так - не будет ни одного
                now_exp_dir_set = set([
                    dir_name for dir_name in os.listdir(project_dir)
                    if os.path.isdir(project_dir + os.sep + dir_name)])
                exp_dir_list = sorted(list(now_exp_dir_set - exp_dir_set))
                if len(exp_dir_list) > 0:
                    last_train_subdir = exp_dir_list[-1]
        else:
            project_dir = ''

        return project_dir + os.sep + last_train_subdir

    def run(self, param_dict, exp_type='val'):

        """
        Запуск эксперимента
        :param param_dict: словарь параметров эксперимента
        :param exp_type: тип эксперимента, один из ключей self.exp_dict (['train', 'val', 'test'])
        :return:
        """

        arg_str = ''.join(
            [' --' + str(k) + ' ' + str(v) for k, v in param_dict.items()]).strip()
        exp_dict = self.exp_dict[exp_type]

        command = 'python ' + self.model_dir + exp_dict['script'] + ' ' + arg_str
        print('command', command)
        # print(exp_dict)
        project_dir = param_dict.get('project',
                                        self.model_dir + exp_dict['def_dir'])

        exp_dict['last_exp_path'] = self.do_experiment(
                                        command, param_dict, project_dir)
        # сейчас нужно обновить или получить реальный param-dict из json
        exp_dict['user_param_dict'] = param_dict
        exp_dict['yolo_param_dict'] = get_data_dict(
            exp_dict['last_exp_path'] + '/opt.yaml')
        exp_dict['history'] = exp_dict.get('history', []) + [exp_dict]
        return

    def count_labels(self, yaml_path):
        """
        Для всех каталогов датасета из описания вывести подсчет рамок классов
        :param yaml_path:
        :return:
        NOTE: если нужны описания конкретной папки нужно пользоватся
         count_classes_in_lab_dir()
        """
        # может быть сделать декоратор TODO для смены рабочего каталога

        wd = os.getcwd()
        os.chdir(self.work_dir)
        data_dict = get_data_dict(yaml_path)
        for k in self.exp_dict.keys():
            img_path = data_dict.get(k)
            if not img_path is None:
                if os.path.isabs(img_path):
                    lab_path = os.path.join(img_path, '../labels/')
                else:
                    lab_path = os.path.join(self.model_dir, img_path, '../labels/')
                lab_path = os.path.abspath(lab_path)
                if os.path.isdir(lab_path):
                    (classes, label_nbr, file_nbr, lab_path
                     ) = count_classes_in_lab_dir(lab_path)
                    print(
                        f'в каталоге {lab_path} найдены описания: {label_nbr} классов: {classes}')
                else:
                    print(f'каталог {lab_path} не найден')
        os.chdir(wd)

    def show_test_images(self,
                         n_samples=5,
                         img_dir=None,
                         #lab_dir=None,
                         ncols=10,
                         img_size=(3,3)):
        """
        Функция для демонстрации нескольких случайных изображений из каталога img_dir
        :param n_samples: общее количество случайных примеров
        :param img_dir: каталог с изображениями
        #:param lab_dir: каталог с аннотациями, пока не используется
        :param ncols: число колонок в выводе
        :param img_size: размер одного изображения
        :return:
        """

        if img_dir is None:
            img_dir = self.exp_dict['test'].get('last_exp_path', None)

        if os.path.isdir(img_dir):
            file_list = [f_name for f_name in os.listdir(img_dir)
                         if (os.path.isfile(img_dir + os.sep + f_name)) &
                         (f_name[-3:].lower() == 'jpg')]
            img_samples = random.sample(file_list, min(n_samples, len(file_list)))
            ncols = min(n_samples, ncols)
            self.show_val_results(img_dir, img_samples, ncols, img_size)
            '''
            for f_name in img_samples:
                print(img_dir + os.sep + f_name)
                with Image.open(img_dir + os.sep + f_name) as img:
                    plt.imshow(img)
                    plt.show()
            '''
        else:
            print('Каталог с изображениями не найден')
        pass

    def show_val_results(self,
                         img_path=None,
                         img_list=('confusion_matrix.png'),
                         ncols=2,
                         img_size=(10, 10)
                         ):
        """
        Метод выводит изображения из списка img_list из каталога  img_path.
        Задумана для вывода графиков, но можно также пользоваться для вывода результатоы
        по конкретным изображениям, например проблемным


        :param img_path: путь для поиска изображений
        :param img_list: список изображений
        :param ncols: число колонок для вывода изображений
        :param img_size:
        :return:
        """


        if not img_path:
            img_path = self.exp_dict['val'].get('last_exp_path',
                                                    get_model_last_exp_dir(self, 'val'))
        print('img_path', img_path)
        if img_path:
            img_f_path_list = [img_path+os.sep+img_f_name for img_f_name in img_list]
            show_list_images(img_f_path_list, ncols=ncols, img_size=img_size)

    def os_com(self, command: str, log_f=True):
        """
        Запуск команды операционной системы
        :param command: текст команды
        :param log_f: если True - перехват вывода
        :return:
        """
        if log_f == True:
            os.system(command + '>>' + self.log_f_name)
        else:
            #!{command}
            os_com(command)
        return

    def save_last_train_weights(self,
                                backup_dir=None,
                                f_last_name=None,
                                # если задано имя  - копируем, иначе - нет
                                f_best_name=None,
                                # если задано имя  - копируем, иначе - нет
                                f_refresh=False):


        def copy_weight(weight_f_name, new_weight_f_name):
            # если None, то этот файл не нужно сохранять
            if new_weight_f_name != None:
                new_weight_path = os.path.join(backup_dir, new_weight_f_name)
                if not f_refresh:
                    new_weight_path = uniquify(new_weight_path)
                source_f_name = os.path.join(last_weights_path, weight_f_name)
                shutil.copy(source_f_name, new_weight_path)
                print(f'Копируем {source_f_name} в {new_weight_path}')

        if (f_last_name == None) and (f_best_name == None):
            print(f'Не выбраны веса для сохранения')
            return
        last_weights_path = os.path.join(
            get_model_last_exp_dir(self, sub='train'), 'weights')
        if os.path.isdir(last_weights_path):
            if backup_dir == None:
                backup_dir = self.backup_dir
                print(f'каталог для сохранения не задан, сохраняем в каталоге /'
                      f'{backup_dir}')
            if not os.path.isdir(backup_dir):
                print(f'каталог для сохранения {backup_dir} не найден,'
                      f' пробуем создать каталог')
                os.makedirs(backup_dir)
            copy_weight('last.pt', f_last_name)
            copy_weight('best.pt', f_best_name)

    def show_bbox_filtered(self,
                           data_yaml,
                           img_dir=None,
                           lab_dir=None,
                           # show_box=True,
                           show_label=0,
                           img_size=(4, 3),
                           class_filter=None,
                           ncols=4,
                           img_nbr=8,
                           n_part=1,  # число частей, на которые разобъем данные
                           part_nbr=None,
                           show_conf=False
                           ):

        data_dict = get_data_dict(data_yaml)
        if not data_dict:
            print(f'описание данных {data_yaml} не найдено или отсутствует')

        cl_nbr = int(data_dict.get('nc', 0))
        name_list = data_dict.get('names', [])

        if (cl_nbr < 1):
            print(f'Количество классов в описании {data_yaml}={cl_nbr}')
        if len(name_list) != cl_nbr:
            print(
                f'Количество имен классов {len(name_list)} не соответствует числу '
                f'классов {cl_nbr}.')
            if show_label > 1:
                print('Имена классов не могут быть выведены')
                show_label = min(show_label, 1)

        # Нужно обработать class_filter может быть список чисел или список строк
        print('class_filter', class_filter)
        try:
            class_filter = [
                name_list.index(x) if type(x) == str else x for x in
                class_filter]
        except:
            print(f'Невозможно применить фильтр {class_filter}'
                  f' для списка классов {name_list}')

        print('class_filter', class_filter)

        def check_dir(img_dir, lab_dir):
            if not img_dir:
                img_dir = data_dict.get('val')
                print(f'Каталог с изображениями не задан, пробуем {img_dir}')
            if not os.path.isabs(img_dir):
                img_dir = os.path.abspath(os.path.join(self.model_dir, img_dir))
            if os.path.isdir(img_dir):
                if not lab_dir:
                    lab_dir = get_lab_img_dir(img_dir)
                    print(f'Каталог с описаниями не задан, пробуем {lab_dir}')
                if not os.path.isdir(lab_dir):
                    print(f'Каталог с описаниями {lab_dir} не существует')
                    return False
            else:
                print(f'Каталог с изображениями {img_dir} не существует')
                return False
            return img_dir, lab_dir

        res = check_dir(img_dir, lab_dir)
        if res:
            img_dir, lab_dir = res
            show_class_images(img_dir,
                              show_box=True,
                              lab_dir=lab_dir,
                              show_label=show_label,
                              name_list=name_list,
                              img_size=img_size,
                              class_filter=class_filter,
                              ncols=ncols,
                              img_nbr=img_nbr,
                              n_part=n_part,
                              # число частей, на которые разобъем данные
                              part_nbr=part_nbr,
                              show_conf=show_conf
                              )
