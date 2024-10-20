import os
#from shutil import copyfile
from pathlib import Path

from .ann_convertors import check_yolov5_yaml, sa,sb
# sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'# /images/, /labels/ substrings
from .ann_convertors import yolov5_img_video_formats, \
    sa, sb, get_ann_format, save_yolo_ann, get_yolov5_ann, ann_ext_dict
from .TerraYolo import save_data_yaml
from .general import fast_scandir, run_fast_scandir



def get_yolo_lab_dir(img_dir):
    return sb.join((str(Path(img_dir)) + os.sep).rsplit(sa, 1))


def get_img_files(img_dir, format_set=frozenset(yolov5_img_video_formats)):
    return [f_name for f_name in os.listdir(img_dir) if
            os.path.splitext(f_name)[1] in format_set]


def explore_img_dir(base_dir_or_list):
    # этим методом возвращаем все словарь, ключи - подкаталоги с изображениями
    # значения число изображений
    if isinstance(base_dir_or_list, (list)):
        subfolders = base_dir_or_list
    else:
        subfolders = fast_scandir(base_dir_or_list)
    result = dict()
    if subfolders:
        for subdir in subfolders:
            subdir_img_files = get_img_files(subdir)
            if subdir_img_files:
                result[subdir] = len(subdir_img_files)
    return result


def check_yolov5_txt_for_img(img_dir, lab_dir, verbose=False):
    if not os.path.isdir(img_dir):
        if verbose:
            print(f'Каталог изображений {img_dir} не найден')
        return ''

    img_f_names = set([os.path.splitext(f_name)[0]
                       for f_name in os.listdir(img_dir)
                       if os.path.splitext(f_name)[1] in yolov5_img_video_formats])
    if len(img_f_names) < 1:
        if verbose:
            print(f'В каталоге {img_dir} не найдены файлы с изображениями')
        return False

    # чтобы получился путь к меткам, и последний каталог должен быть строго /images/
    # просто /images без слэша в конце будет ошибка. Если не последний - все равно,

    if not os.path.isdir(lab_dir):
        if verbose:
            print(f'Каталог меток {lab_dir} не найден')
        return False

    lab_f_names = set([os.path.splitext(f_name)[0]
                       for f_name in os.listdir(lab_dir)
                       if os.path.splitext(f_name)[1] == '.txt'])

    if len(img_f_names & lab_f_names) < 1:
        if verbose:
            print(f'В каталоге {img_dir} не найдены подходящие аннотации')
        return False

    return True


def check_yolov5_img_dir_name(img_dir):
    if os.path.normpath(img_dir) == os.path.normpath(get_yolo_lab_dir(img_dir)):
        return False
    return True


def img_dir_name_analyze(img_dir_dict, f_name_prior=True):
    # сначала проверяем, что потенциальное имя каталога с изображением годится для yolov5
    # строим список каталогов с метками и если совпадают, то не годится
    lab_dir_list = []
    for img_dir in img_dir_dict.keys():
        lab_dir_list.append(get_yolo_lab_dir(str(img_dir) + os.sep))
        '''           
        if os.path.isdir(lab_dir):
            lab_files = [f_name for f_name in lab_dir if os.path.splitext(f_name)=='.txt']
            lab_dir_dict[lab_dir]=len(lab_files)
        else:
            lab_dir_dict[lab_dir]=0
        '''
    bad_name_img_dir = [img_dir for ind, img_dir in enumerate(img_dir_dict.keys()) if
                        img_dir == lab_dir_list[ind]]
    for img_dir in bad_name_img_dir:
        print(
            f'Каталог с изображениями {img_dir} не содержит в пути стандартной папки {sa}')
        # delete img_dir_dict[img_dir]

    img_train_dir, img_val_dir, img_test_dir = '', '', ''
    if len(img_dir_dict) < 1:
        print('Не найдены каталоги с изображениями')
        return img_train_dir, img_val_dir, img_test_dir

    # ищем среди имен каталогов train, test, val(valid)
    train_dict = dict()
    val_dict = dict()
    test_dict = dict()

    if f_name_prior == True:
        if len(img_dir_dict) > 1:
            for img_dir in img_dir_dict.keys():
                path_parts_reversed = Path(img_dir).resolve().parts[::-1]
                # если среди каталогов есть 'train'
                if 'train' in path_parts_reversed:
                    # добавляем в соотвествующий словарь макс уровень на котором нашли train
                    # максимальный на случай train может встретиться в пути и не один раз
                    # train и другие ключи типа val тоже могут встретиться в пути
                    train_dict[img_dir] = len(path_parts_reversed
                                              ) - path_parts_reversed.index('train')

                # val
                val_pos = -1
                if 'val' in path_parts_reversed:
                    val_pos = len(path_parts_reversed
                                  ) - path_parts_reversed.index('val')

                if 'valid' in path_parts_reversed:
                    # добавляем в соотвествующий словарь максимальный уровень
                    val_pos = max(len(path_parts_reversed
                                      ) - path_parts_reversed.index('valid'), val_pos)

                if val_pos >= 0:
                    val_dict[img_dir] = val_pos
                # test

                if 'test' in path_parts_reversed:
                    # добавляем в соотвествующий словарь максимальный уровень
                    test_dict[img_dir] = len(path_parts_reversed
                                             ) - path_parts_reversed.index('test')

            def get_dir_from_max_level(dir_dict):
                max_dir_level = max(dir_dict.values())
                max_dir_ind = list(dir_dict.values()).index(max_dir_level)
                img_dir = list(dir_dict.keys())[max_dir_ind]
                return img_dir

            # если f_name_prior==True и в каталогах с изображениями встретился каталог 'train'
            if train_dict:
                img_train_dir = get_dir_from_max_level(train_dict)

                if img_train_dir in val_dict.keys():
                    del val_dict[img_train_dir]
                if img_train_dir in test_dict.keys():
                    del test_dict[img_train_dir]

            # если train не найден, то остальное ('val', 'test') игнорируем
            if val_dict:
                img_val_dir = get_dir_from_max_level(val_dict)
                if img_val_dir in test_dict.keys():
                    del test_dict[yaml_dict['val']]

            if test_dict:
                img_test_dir = get_dir_from_max_level(test_dict)

    sorted_img_dir_dict = dict(sorted(img_dir_dict.items(), key=lambda item: item[1]))

    # если не находим по названиям - считаем картинки и в img_train_dir назначаем
    # каталог с max количеством изображений
    if not img_train_dir:
        img_train_dir = list(sorted_img_dir_dict.keys())[0]
        del sorted_img_dir_dict[img_train_dir]

    if not img_val_dir:
        if sorted_img_dir_dict:
            img_val_dir = list(sorted_img_dir_dict.keys())[0]
            del sorted_img_dir_dict[img_val_dir]

    # в случае подсчета по количеству изображений test не ищем
    # if not img_test_dir:

    if (img_test_dir != '') & (img_val_dir == ''):
        img_val_dir = img_test_dir

    if (img_val_dir != '') & (img_train_dir == ''):
        img_train_dir = img_val_dir

    if (img_train_dir != '') & (img_val_dir == ''):
        img_val_dir = img_train_dir

    return img_train_dir, img_val_dir, img_test_dir

### общие соображения
# класс привязан к структуре каталогов и содержит от одного подраздела подкатлога
# yaml для каталогов имеет вспомогательное значение, для классов - основное
# смысл в том, что класс для того, чтобы получить yaml,
# если yaml есть - то и класс не так чтоб нужен
# ну еще как развитие - хранить результаты в классе, но это не точно
# train - val - test - все остальное misc, в общем случае
# привязываемся к именам - train , val-valid, test
# нет train с аннотациями - нет датасета
# потому если нет имени train - выбираем для train самый большой каталог
# в test может не быть аннотаций
# у val-valid должны быть аннотации, потому, если нет имени val-valid
# но есть test и для test есть аннотации - test становиться val-valid
# если testa нет или нет для него аннотаций - train становится val
#


class YoloV5Dataset:
    def __init__(self,
                 yaml_path_dir=None,
                 train_img_subdir='', val_img_subdir='', test_img_subdir='',
                 # train_lab_dir='', val_lab_dir='', test_lab_dir='',
                 # data_root_path=None,
                 # cl_labels=None,
                 names=(),
                 new_yaml_path=None, new_yaml_f_name='dataset.yaml',
                 f_autosave=False, f_overwrite=False,
                 ):

        if not os.path.exists(yaml_path_dir):
            print(f'файл/каталог {yaml_path_dir} с данными не найден')
            return None

        self.data_keys = ['train', 'val', 'test']
        self.dest_key = 'yolov5'
        self.data_dict = dict()
        self.yaml_dict = dict()

        self.new_yaml_path = new_yaml_path
        self.new_yaml_f_name = new_yaml_f_name
        self.f_autosave = f_autosave
        self.f_overwrite = f_overwrite

        for k in self.data_keys:
            self.data_dict[k] = dict()

        # на этом шаге ищем yaml описания
        if os.path.isfile(yaml_path_dir):
            self.base_dir = os.path.split(yaml_path_dir)
            self.parse_yaml(yaml_path=yaml_path_dir)
        elif os.path.isdir(yaml_path_dir):
            self.base_dir = yaml_path_dir
            # прочие yaml будем искать только при отсутствии списка классов
            if len(names) < 1:
                self.explore_base_dir_yaml()
            else:
                print('дан список классов, пропускаем проверку yaml - файлов')

        # если дошли до сюда и yaml_dict значит найден подходящий yaml можно уходить
        if self.yaml_dict:
            return

        print(
            f'По адресу {yaml_path_dir} не найдены подходящие yaml-описания датасетов, продолжаем анализ')

        if train_img_subdir:
            train_img_dir = os.path.join(self.base_dir, train_img_subdir)
            if not check_yolov5_img_dir_name(train_img_dir):
                print(
                    f'Имя {train_img_dir} должно содержать {sa} и не может быть каталогом изображений yolov5')
                return None
            if val_img_subdir:
                val_img_dir = os.path.join(self.base_dir, val_img_subdir)
            if test_img_subdir:
                test_img_dir = os.path.join(self.base_dir, test_img_subdir)
        else:
            train_img_dir, val_img_subdir = None, None

        if len(names) > 0:
            self.names = names
            self.nc = len(self.names)
            if not train_img_dir:
                # в этом случае будут проверены имена подкаталогов
                train_img_dir, val_img_dir, test_img_dir = self.get_img_dir_list()
                if self.yaml_dict:
                    return
            for img_dir in [train_img_dir, val_img_dir]:
                if not check_yolov5_txt_for_img(img_dir, get_yolo_lab_dir(img_dir)):
                    del self.names
                    del self.nc
                    return None
                tmp_dict = dict()
                tmp_dict['train'] = train_img_dir + os.sep
                tmp_dict['val'] = val_img_dir + os.sep
                if test_img_dir: tmp_dict['test'] = test_img_dir + os.sep
                tmp_dict['path'] = self.base_dir
                tmp_dict['names'] = self.names
                tmp_dict['nc'] = self.nc
                # yaml_path
                self.set_class_param(tmp_dict, self.base_dir, yaml_path=None)
                # self.save_yaml()
                return

        # len(names)==0,
        img_dict = explore_img_dir(self.base_dir)
        train_img_dir, val_img_dir, test_img_dir = img_dir_name_analyze(img_dict,
                                                                        f_name_prior=True)
        if not train_img_dir:
            print(f'В каталоге {self.base_dir} не найдены каталоги с изображениями')
            return None
        # если дошли до сюда, то остается искать только аннотации других форматов

        format_founded = get_ann_format(self.base_dir)
        # format_founded = set(format_founded)-set([self.dest_key])
        if len(format_founded) > 0:
            print(f'Найдены аннотации следующих форматов: {format_founded}')
            for f_key in format_founded:
                if f_key != self.dest_key:
                    # остальные форматы не рассматриваем, берем только первый подходящий
                    if self.get_labels(train_img_dir, val_img_dir, test_img_dir, f_key):
                        return
        else:
            print(f'в каталоге {yaml_path_dir} не найдены подходящие аннотации')
        return None

    def parse_yaml(self, yaml_path):
        print('parse_yaml', yaml_path)
        yaml_dict, base_dir = check_yolov5_yaml(yaml_path)
        print('yaml_dict, base_dir', base_dir)
        if yaml_dict:
            print('parse_yaml set_class_param')
            self.set_class_param(yaml_dict, self.base_dir, yaml_path)

    def get_img_dir_list(self):
        # если заданы каталоги train_img_dir, val_img_dir, test_img_dir они будут в приоритете
        # возможны ситуации нет каталога - нет изображений -нет меток-

        # img_lab_dir_dict = dict(zip([train_img_dir, val_img_dir, test_img_dir],
        # [train_lab_dir, val_lab_dir, test_lab_dir]))

        def check_tmp_yaml():
            def get_key_dir(k, img_dir):
                if os.path.isdir(img_dir):
                    img_list = get_img_files(img_dir)
                    if len(img_list) > 0:
                        tmp_yaml[k] = img_dir

            tmp_yaml = dict()

            get_key_dir('train', train_img_dir)
            get_key_dir('val', val_img_dir)

            if not tmp_yaml.get('train', ''):
                tmp_yaml['train'] = tmp_yaml.get('val', '')

            if tmp_yaml.get('train', '') != '':
                get_key_dir('test', test_img_dir)
                tmp_yaml['names'] = self.names
                tmp_yaml['nc'] = self.nc

                yaml_dict, yaml_base_dir = check_yolov5_yaml(tmp_yaml,
                                                             base_dir=self.base_dir)
                if yaml_dict:
                    print(
                        f'Словарь {yaml_dict}, прошел проверку и установлен как основной')
                    self.set_class_param(yaml_dict, yaml_base_dir, yaml_path=None)
                    return True
            return False

        '''                
        if check_tmp_yaml():
            return train_img_dir, val_img_dir, test_img_dir
        '''

        img_dict = explore_img_dir(self.base_dir)
        # print('get_img_dir_list img_dict', img_dict)
        train_img_dir, val_img_dir, test_img_dir = img_dir_name_analyze(img_dict,
                                                                        f_name_prior=True)
        # print('get_img_dir_list img_dict', train_img_dir, val_img_dir, test_img_dir)

        if check_tmp_yaml():
            return train_img_dir, val_img_dir, test_img_dir
        return
        '''
        # если дошли сюда возможен вариант когда аннотации заданы в других каталогах
        # здесь мы проверяем аннотации yolo, другие форматы - пока пропускаем
        for img_dir in [train_img_dir, val_img_dir, test_img_dir]:
            if check_yolov5_txt_for_img(img_dir, img_lab_dir_dict[img_dir]):
                new_lab_dir = get_yolo_lab_dir(img_dir)
                os.makedirs(new_lab_dir, exist_ok=True)
                source_dir = img_lab_dir_dict[img_dir]
                print(f'копируем аннотации из каталога {source_dir} в {new_lab_dir}')
                for f_name in os.listdir(source_dir):
                    copyfile(os.path.join(source_dir, f_name), new_lab_dir)
                continue
            else: return train_img_dir, val_img_dir, test_img_dir

        #возможно удалось скопировать новые метки, проверяем в третий раз
        check_tmp_yaml()
        return train_img_dir, val_img_dir, test_img_dir
        '''

    def save_yaml(self, yaml_path='', yaml_file='', f_overwrite=False):
        if self.f_autosave:
            if not yaml_path:
                yaml_path = self.new_yaml_path if self.new_yaml_path else self.base_dir
            if not yaml_file:
                yaml_file = self.new_yaml_f_name
            f_overwrite = self.f_overwrite
        dest_yaml_path = os.path.join(yaml_path, yaml_file)
        if dest_yaml_path:
            print(f'сохранен новый словарь {dest_yaml_path}')
            save_data_yaml(self.yaml_dict, dest_yaml_path, f_overwrite)

    def set_class_param(self, yaml_dict, base_dir, yaml_path=None):
        # смысл такой, что path может и не быть, а base_dir можно вычислить или задать
        # base_dir - это каталог path из yaml, если такой есть либо каталог
        if isinstance(yaml_path, (dict)):
            self.yaml_path = None
        else:
            self.yaml_path = yaml_path
        self.base_dir = base_dir
        self.yaml_dict = yaml_dict
        self.save_yaml()

    def explore_base_dir_yaml(self):
        # если пришел каталог, то надо  первую очередь искать yaml
        # потому что без него нет информации по классам
        yaml_list = [f_name for f_name in os.listdir(self.base_dir) if
                     os.path.splitext(f_name)[1] == '.yaml']
        if yaml_list:
            print(f' В каталоге {self.base_dir}, найдены файлы описаний: {yaml_list}')
            for yaml_f_name in yaml_list:
                yaml_path = os.path.join(self.base_dir, yaml_f_name)
                yaml_dict, yaml_base_dir = check_yolov5_yaml(yaml_path)
                if yaml_dict:
                    print(
                        f'Словарь {yaml_f_name}, прошел проверку и установлен как основной')
                    self.set_class_param(yaml_dict, yaml_base_dir, yaml_path)
                    return
                else:
                    print(f'Словарь {yaml_f_name}, не прошел проверку')
        # если сюда дошли, значит подходящие yaml не обнаружены, нужно анализировать изображения

    def get_labels(self, train_img_dir, val_img_dir, test_img_dir, format_key):
        # ann_ext_dict = {'yolov5': '.txt', 'coco': '.json', 'pascal_voc': '.xml'}
        ann_f_path_list = run_fast_scandir(self.base_dir, ann_ext_dict[format_key])
        if format_key == 'coco':
            # в одном файле много меток, берем список файлов
            ann_f_path_list = ann_f_path_list[1]
        elif format_key == 'pascal_voc':
            # в одном каталоге много меток, берем список каталогов
            ann_f_path_list = ann_f_path_list[0]
        # print(format_key, ann_f_path_list)
        tot_class_dict, tot_box_dict = dict(), dict()
        for ann_path in ann_f_path_list:
            # class_dict, img_dict, box_dict
            class_dict, _, box_dict = get_yolov5_ann(ann_path, input_format=format_key)
            tot_class_dict.update(class_dict)
            tot_box_dict.update(box_dict)

        # print(tot_class_dict)
        # print(len(tot_box_dict))
        tmp_dict = dict()
        if len(tot_class_dict) < 1:
            print(f'В {ann_f_path_list} не найдены описания классов')
            return False
        else:
            tmp_dict['names'] = list(tot_class_dict.values())
            tmp_dict['nc'] = len(tmp_dict['names'])

        def save_labels(img_dir, f_save=self.f_autosave):
            img_f_list = get_img_files(img_dir)
            dir_img_boxes = {k: v for k, v in tot_box_dict.items() if k in img_f_list}
            ann_nbr = len(dir_img_boxes)
            if ann_nbr > 0:
                lab_dir = get_yolo_lab_dir(img_dir)
                if f_save:
                    if not os.path.isdir(lab_dir):
                        print(f'каталог {lab_dir} не найден, создаем')
                        os.makedirs(lab_dir)
                    save_yolo_ann(lab_dir, dir_img_boxes)
                else:
                    print(
                        f'Для изображений из каталога {img_dir} найдены, но не сохранены {ann_nbr} аннотаций. Параметр f_autosave не установлен')

            else:
                print(f'Не найдены аннотации для изображений из каталога {img_dir}')

        for img_dir in set([train_img_dir, val_img_dir, test_img_dir]):
            if img_dir:
                save_labels(img_dir)

        tmp_dict['train'] = train_img_dir + os.sep
        tmp_dict['val'] = val_img_dir + os.sep
        if test_img_dir: tmp_dict['test'] = test_img_dir + os.sep
        tmp_dict['path'] = self.base_dir

        yaml_dict, _ = check_yolov5_yaml(tmp_dict, base_dir=self.base_dir)
        if yaml_dict:
            self.set_class_param(yaml_dict, self.base_dir, yaml_path=None)
            return True

        return False

# yolo_v5_dataset = YoloV5Dataset(yaml_path_dir=r'C:/WORK/NU/YOLO/data/10K/')
# yolo_v5_dataset = YoloV5Dataset(yaml_path_dir=r'C:/WORK/NU/YOLO/data/10K_no_yaml/')
# yolo_v5_dataset = YoloV5Dataset(yaml_path_dir=r'C:/WORK/NU/YOLO/data/10K_no_yaml/', names=['варон', 'лисиц'])
# yolo_v5_dataset = YoloV5Dataset(yaml_path_dir=r'C:/WORK/NU/YOLO/data/10K_no_yaml/', names=['варон', 'лисиц'])
# yolo_v5_dataset = YoloV5Dataset(yaml_path_dir=r'C:/WORK/NU/YOLO/data/BCCD_coco/')#, names=['варон', 'лисиц'])
# yolo_v5_dataset = YoloV5Dataset(yaml_path_dir=r'C:/WORK/NU/YOLO/data/BCCD_images/')