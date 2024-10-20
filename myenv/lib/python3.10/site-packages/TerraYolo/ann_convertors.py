import os
import json
import numpy as np
import xml.etree.ElementTree as ET
import glob
from pathlib import Path
from .TerraYolo import get_data_dict, save_data_yaml



yolov5_img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']
yolov5_vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']

format_keys = ['yolov5', 'coco', 'pascal_voc']
yolov5_img_video_formats = ['.' + ext for ext in
                            yolov5_img_formats + yolov5_vid_formats]

sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings

ann_templ_dict = {
    'yolov5': '**/*.txt', 'coco': '**/*.json', 'pascal_voc': '**/*.xml'}

ann_ext_dict = {
    'yolov5': '.txt', 'coco': '.json', 'pascal_voc': '.xml'}


def xywh2xywhn_uno(x, w=640, h=640):
    # def get_yolov5_box(img_size, box):

    # COCO "bbox" : [x,y,width,height]
    # x = (x_min, y_min, wth, ht) координаты левого верхнего угла рамки в пикселях

    # img_wth, img_ht = img_size
    # x_min, y_min, wth, ht = box

    y = x.copy()

    y[2] = y[2] / w  # меняем на относительные значения ширину
    y[3] = y[3] / h  # и высоту рамки
    y[0] = y[0] / w + y[2] / 2  # и рассчитываем координаты центра рамки
    y[1] = y[1] / h + y[3] / 2

    return y


def xyxy2xywhn_uno(x, w=640, h=640):
    # VOC PASCAL "bboxes" : [xmin,ymin, xmax, ymax]
    # x = (xmin,ymin, xmax, ymax) координаты левого верхнего и правого нижнего угла рамки в пикселях

    y = x.copy()
    y[2] = (y[2] - y[0]) / w  # относительная ширина
    y[3] = (y[3] - y[1]) / h  # относительная высота рамки
    y[0] = y[0] / w + y[2] / 2  # и рассчитываем координаты центра рамки
    y[1] = y[1] / h + y[3] / 2

    return y


def get_yolo_ann_from_coco(ann_f_path):
    """

    :param ann_f_path: путь к файлу аннотаций coco
    :return:
        class_dict - словарь классов - ключ - номер класса, значение - наименование
        img_dict - словарь параметров изображения, имя файла - ключ,
                значение - список [ширина, высота]
        box_dict - параметры файла, ключ - имя файла, значение - массивы
                [cl, xn, yn, wn, hn, ... может быть что угодно, здесь не провреяется]
    """

    assert os.path.isfile(ann_f_path) == True

    # img_dir = os.path.split(ann_f_path)[0]

    with open(ann_f_path) as cocojson:
        annotations_json = json.load(cocojson)

    class_dict = {cl['id']: cl['name'] for cl in annotations_json['categories']}

    # пока так, возможно понадобится cat['path'] + cat['file_name']
    # print(annotations_json['images'][0].keys(), annotations_json['images'][0])
    img_dict = {img['file_name']: [int(img['width']), int(img['height'])]
                for img in annotations_json['images']}
    # tmp_dict нужен чтоб связать имя файла с id в аннотациях
    tmp_dict = {img['id']: img['file_name'] for img in annotations_json['images']}

    # получаем рамки классов в формате coco
    box_dict = dict([])
    for ann in annotations_json['annotations']:
        if ann.get('bbox'):  # в COCO описании 'bbox' может и не быть
            img_f_name = tmp_dict[ann['image_id']]
            box_dict[img_f_name] = box_dict.get(img_f_name, []) + [
                [ann['category_id']] +
                xywh2xywhn_uno(ann['bbox'], w=img_dict[img_f_name][0],
                               h=img_dict[img_f_name][1])]

    return class_dict, img_dict, box_dict


def get_yolo_ann_from_p_voc(source_path):
    # def vocann2dict(source_path):

    # функция вернет список словарей с именами файлов, размерами и координатами

    # https://blog.paperspace.com/train-yolov5-custom-data/
    # Function to get the data from XML Annotation
    def extract_info_from_xml(xml_file):
        root = ET.parse(xml_file).getroot()

        # Initialise the info dict
        info_dict = {}
        info_dict['bboxes'] = []

        # Parse the XML Tree
        for elem in root:
            # Get the file name
            if elem.tag == "filename":
                info_dict['filename'] = elem.text

            # Get the image size
            elif elem.tag == "size":
                image_size = []
                for subelem in elem:
                    image_size.append(int(subelem.text))

                info_dict['image_size'] = tuple(image_size)

            # Get details of the bounding box
            elif elem.tag == "object":
                bbox = {}
                for subelem in elem:
                    if subelem.tag == "name":
                        bbox["class"] = subelem.text

                    elif subelem.tag == "bndbox":
                        for subsubelem in subelem:
                            bbox[subsubelem.tag] = int(subsubelem.text)
                info_dict['bboxes'].append(bbox)

        return info_dict

    ann_info_dict_list = []
    for f_name in os.listdir(source_path):
        # не проверяются посторонние файлы, будет ошибка
        if os.path.splitext(f_name)[1] != ann_ext_dict['pascal_voc']:
            continue
        f_path = os.path.join(source_path, f_name)
        ann_info_dict_list.append(extract_info_from_xml(f_path))

    img_dict = {img['filename']: img['image_size']
                for img in ann_info_dict_list}  # ['images']

    class_list = []
    box_dict = dict()
    for ann_dict in ann_info_dict_list:
        for box_dict in ann_dict["bboxes"]:
            class_list.append(box_dict['class'])

    class_list = sorted(list(set(class_list)))
    class_dict = {k: v for k, v in enumerate(class_list)}
    class_dict_name_nbr = {v: k for k, v in class_dict.items()}

    box_dict = dict()

    for ann_dict in ann_info_dict_list:
        f_name = ann_dict['filename']
        img_box_list = []
        for tmp_dict in ann_dict["bboxes"]:
            img_box_list.append([class_dict_name_nbr[tmp_dict['class']]] +
                                xyxy2xywhn_uno([tmp_dict['xmin'], tmp_dict['ymin'],
                                                tmp_dict['xmax'], tmp_dict['ymax']],
                                               # width,               height
                                               img_dict[f_name][0], img_dict[f_name][1])
                                )
        box_dict[f_name] = img_box_list

    return class_dict, img_dict, box_dict


# input_formats
get_bbox_from_misc_format = {'coco': get_yolo_ann_from_coco,
                             'pascal_voc': get_yolo_ann_from_p_voc}


def save_yolo_ann(dest_path, box_dict):
    for img_f_name, box_list in box_dict.items():
        lab_f_name = '.'.join(img_f_name.split('.')[:-1]) + '.txt'
        lab_f_path = os.path.join(dest_path, lab_f_name)
        with open(lab_f_path, 'w') as f:
            f.write('\n'.join(
                [' '.join(list(map(str, box))) for box in box_list]))


# def create_yolov5_yaml(class_dict,

def get_yolov5_ann(source_path, input_format='coco'):
    return get_bbox_from_misc_format[input_format](source_path)

def convert_2_yolov5_ann(source_path, dest_path, input_format='coco'):
    class_dict, _, box_dict = get_bbox_from_misc_format[input_format](
        source_path)
    if box_dict:
        if not os.path.isdir(dest_path):
            print(f'каталог {dest_path} не найден, создаем')
            os.makedirs(dest_path)
        save_yolo_ann(dest_path, box_dict)

    return class_dict


'''
def is_coco(ann_dir_file_path):
    #здесь сверяем является ли файл метками coco
    def is_coco_label_correct();
        # а здесь нужно проверить корректны ли сами значения 
        #(наприемр не больше размеров изображения и тп)
        pass

    #Not Implemented
    res = False
    return res


def check_ann_format(ann_file):
    # xyxy xywh xywhn xyxyxn
    pass

def is_yolov5(file_path):
    # тут какой-нибудь pydantic подошел бы
    # нам 

def get_img_json_names(ann_dir):
    img_names = [os.path.splitext(f_name)[0] for f_name in os.listdir(ann_dir)
                        if os.path.splitext(f_name)[1] in yolov5_img_video_formats]
    lab_names = [os.path.splitext(f_name)[0]
                        for f_name in glob.glob(os.path.join(ann_dir, '*.json'))]
    return img_names, lab_names

'''


def is_yolov5_ann(ann_path_list, format_type='detection', f_all_files=False):
    def is_detection(f_path):
        # схему можно расширять, пока проверяем возможность преобразовать типы
        try:
            with open(f_path, 'r') as f:
                labels = np.array(
                    [x.split() for x in f.read().strip().splitlines()],
                    dtype=np.float32)  # labels
            return labels.shape[0]  # число строк
        except:
            # print('ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!')
            return False

    # print(ann_path_list)
    format_check = {'detection': is_detection}
    checker = format_check[format_type]
    res = []

    if f_all_files:
        res = [f_path for f_path in ann_path_list if checker(f_path)]
    else:
        for f_path in ann_path_list:
            if checker(f_path):
                return (f_path)

    return res


def is_coco_ann(ann_path_list, format_type='detection', f_all_files=False):
    def is_coco_data(f_path, coco_key):
        with open(f_path, 'r') as cocojson:
            annotations_json = json.load(cocojson)

        if annotations_json.get('annotations'):
            for img_ann in annotations_json['annotations']:
                if img_ann.get(coco_key):
                    return True
        return False

    coco_keys_dict = {'detection': 'bbox', 'segmentation': 'segmentation'}
    coco_key = coco_keys_dict[format_type]

    res = []
    if f_all_files:
        res = [f_path for f_path in ann_path_list if
               is_coco_data(f_path, coco_key)]
    else:
        for f_path in ann_path_list:
            if is_coco_data(f_path, coco_key):
                return (f_path)
    return res


def is_pascal_voc_ann(ann_path_list, format_type='detection', f_all_files=False):
    def is_detection(f_path):
        xml_root = ET.parse(f_path).getroot()
        # Parse the XML Tree
        for elem in xml_root:
            if elem.tag == "object":
                bbox = []
                for subelem in elem:
                    # if subelem.tag == "name":
                    # bbox["class"] = subelem.text
                    if subelem.tag == "bndbox":
                        for subsubelem in subelem:
                            try:
                                bbox.append(int(subsubelem.text))
                            except:
                                continue
                    if len(bbox) >= 4:
                        return True
        return False

    format_check = {'detection': is_detection}
    checker = format_check[format_type]
    res = []

    if f_all_files:
        res = [f_path for f_path in ann_path_list if checker(f_path)]
    else:
        for f_path in ann_path_list:
            if checker(f_path):
                return (f_path)
    return res


format_ann_check_dict = dict(zip(['yolov5', 'coco', 'pascal_voc'],
                                 [is_yolov5_ann, is_coco_ann, is_pascal_voc_ann]))

def check_ann_format(ann_path_list,
                     format_key, format_type='detection', f_all_files=False):
    # вынес отдельной функцией на случай проверки формата по известным ключам
    return format_ann_check_dict[format_key](ann_path_list,
                                             format_type=format_type,
                                             f_all_files=f_all_files)


def get_ann_format(ann_dir_file_path,
                   format_type='detection', f_all_formats=False):
    """

    :param ann_dir_file_path:
    2 варианта либо каталаг - yolov5, PASCAL_VOC, COCO
    либо файл  COCO, yolov5
    в первом приближении такой алгоритм:
    - если каталог и >=1 txt файла - yolov5
    - если каталог json файлы и соответствующие им изображения - pascal_voc
    - если каталог и >1 json файлов - но неет соответствующих изображений - coco

    в следующих уточнениях можно проверять собственно форматы файлов аннотаций

    - если файл
        - если json - COCO
        - если yaml - yolo + поиск/проверка путей


    :return:
        возвращает 'yolov5' , 'coco', 'pascal_voc' или False
    """

    '''
    def check_yolov5_lab(ann_dir, format_key= f_key, format_type='detection'):

        # работает но очень долго и непонятно ради чего
        #return format_ann_check_dict[f_key](probably_ann_list,
                                            #format_key = f_key,
                                            #format_type='detection')
        return check_ann_format(probably_ann_list, format_key= f_key,
                                                    format_type=format_type)
    '''
    # templ_dict = dict(zip(['yolov5', 'coco', 'pascal_voc'],
    #                            ['**/*.txt', '**/*.json', '**/*.xml']))

    # порядок ключей  имеет значение - сначала нужно проверить что нет описания yolo
    res = []
    if os.path.isdir(ann_dir_file_path):
        for f_key in format_ann_check_dict.keys():
            probably_ann_list = glob.glob(os.path.join(ann_dir_file_path,
                                                       ann_templ_dict[f_key]),
                                          recursive=True)
            # print(probably_ann_list)
            if check_ann_format(probably_ann_list, f_key,
                                format_type=format_type, f_all_files=False):
                if f_all_formats:
                    res.append(f_key)
                    continue
                else:
                    return [f_key]

    elif os.path.isfile(ann_dir_file_path):
        # Not implemented yet
        # пока задача стоит анализировать директорию
        pass
    pass

    return res


# в yolov5 в описании датасета указывается только пути к ИЗОБРАЖЕНИЯМ
# ищем всех родителей и  на уровне каталога images создаем каталог labels
# если images не находим - создаем на уровне родительского
# и, с учетом флага f_replace копируем в этот каталог метки изображений
# тут тоже возможны варианты типа родительский каталог может быть общим для
# нескольких дргуих и тогда смешаются метки разных катлогов в один

# 1. найдем все картинки
# 2. найдем все каталоги картинок
# 3. каждoму каталогк сопоставим каталог меток
# 4. далее по списку - создаем каталог меток - создаем в нем метки соответствующих изображений
# 5. мб все через pandas

# это заготовка для автоматического создания структуры каталогов
# этот код не безопасен так как не проверяются каталоги из списка
# на вложенность, потому переименование одного может поменять имена других
'''
def change_structure(img_dir_list, sa, f_rename=False):#, sb
    for dir_d in img_dir_list:
        # len(dir_d.rsplit(sa, 1)) если >1, то нужная подстрока sa содержится 
        # в пути и каталог dir_d переделывать не нужно, пропускаем
        if len(dir_d.rsplit(sa, 1))==1: 
            # запоминаем содержимое каталога
            # тут есть варианты запоминать ли только картинки или все файлы
            # пока не заморачиваемся и запомнинаем все
            f_name_list=os.listdir(dir_d)

            #new_img_dir = os.path.join(dir_d, sa)
            new_img_dir = dir_d + sa# path.join не работает тк sa ограничена os.sep = пути из корневого каталога
            print(new_img_dir)
            os.makedirs(new_img_dir, exist_ok=True)
            # имя нового каталога нужно исключить из списка копируемых файлов, 
            # на случай, если оно там было
            new_dir_name = os.path.split(new_img_dir)[1]
            if new_dir_name in f_name_list: 
                                            f_name_list.remove()

            print(f'Перемещаем {len(f_name_list)} файлов из {dir_d} в {new_img_dir}')
            for f_name in f_name_list:
                shutil.move(os.path.join(dir_d, f_name), new_img_dir)
        else:
            print(f'не меняем каталог {dir_d}')
        #break
'''


def get_yolov5_images_glob(src_img_path, f_change_dir_structure=False,
                           f_rename=False):
    # брак

    # это заготовка для автоматического создания структуры каталогов
    # не доделано
    '''
    def del_parent_dir(dir_list):
        def has_parent():

        for fir img_path_list
    '''

    img_path_list = [
        f_path for f_path in glob.glob(os.path.join(src_img_path, '**/*'),
                                       recursive=True)
        if os.path.splitext(f_path)[1] in yolov5_img_video_formats]

    img_dir_list = sorted(set([os.path.split(f_path)[0]
                               for f_path in img_path_list]))
    # если каталог кончается на images, os.path.split(f_path)[0] вернет
    # '... /images' на конце, а нужно '... /images/' потому
    img_dir_list = [img_dir + os.sep for img_dir in img_dir_list]
    # print(src_img_path, img_dir_list)
    # заготовка для автоматического создания структуры каталогов не доделано
    '''
    # в зависимости от того, что мы получили на вход
    # пропускаем вложенные каталоги среди найденных, для минимизации 
    # img_dir_list = del_parent_dir(img_dir_list):
    '''

    # то что ниже навязано кодом из yolov5 utils/dataloaders.py
    '''
    def img2label_paths(img_paths):
        # Define label paths as a function of image paths
        sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
        return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]
    '''
    # нужно подготовить структуру каталогов для аннотаций,
    # соответствующую структуре для изображений

    # sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    # sa, sb = 'nu_yolo', '/labels/'
    yolov5_lab_dir_list = [sb.join(x.rsplit(sa, 1)) for x in img_dir_list]

    # если путь к изображениям == путь к аннотациям - это означает, что нужной
    # подстроки sa ('/images/') не было в имени img_dir и нужно править
    # структуру каталогов,  иначе yolov5 не заведется

    bad_img_dir_idx = [i for i in range(len(img_dir_list))
                       if img_dir_list[i] == yolov5_lab_dir_list[i]]
    if bad_img_dir_idx:
        # дальнейшие варианты очень мудреные, предложим переименовать каталог
        print(f'Пути к изображениям yolov5 должны содержать подстроку {sa}')
        # возможны варианты когда некоторые пути называются правильно
        # но, чтобы не усложнять их рассматривать не будем
        return [], [], [], []

    # Not Implemented - не доделано
    # заготовка для автоматического создания структуры каталогов
    # правка разрешена, если выставлен флаг f_change_dir_structure
    '''
    # вкратце, если понадобится автоматическое переименование, то
    # нужно будет найти все 'плохие'  каталоги
    # проанализировать их на вложенность и или делать для всех каталогов 
    # верхнего уровня них каталог labels (и он может оказаться общим для 
    # нескольких)
    # или поднимась к корню делать отдельный для каждого
    # и среди них некоторые тоже могут стать общими

    for ind, img_dir in enumerate(img_dir_list):
        if img_dir == lab_dir_list[ind]:
            print(f'Путь к каталогам изображений yolov5 не содержит подстроку {sa}')
            if f_change_dir_structure:
                print('Адаптируем структуру каталогов....')
                change_structure(img_dir_list, sa, f_rename=f_rename)
                #change_structure(img_dir_list, 'nu_yolo')

            else:
                lab_dir_list[ind]=''
            return []
    '''

    ann_ext_list = [os.path.splitext(v)[1] for v in ann_templ_dict.values()]
    ann_path_list = [
        f_path for f_path in glob.glob(os.path.join(src_img_path, '**/*'),
                                       recursive=True)
        if os.path.splitext(f_path)[1] in ann_ext_list]

    # каталоги из списка img_dir_list существуют реально
    # каталоги из списка lab_dir_list могут существовать, а могут и нет
    return img_path_list, img_dir_list, yolov5_lab_dir_list, ann_path_list


def create_yolov5_yaml(data_dir,
                       img_path_list, img_dir_list, lab_dir_list,  # probably_ann_list,
                       dest_yaml_path,
                       class_dict,
                       yaml_f_name_path='dataset.yaml',
                       yaml_path_param=None,  # ключ path
                       f_overwrite=False):
    yaml_dict = dict()

    # ищем среди имен каталогов train, test, val(valid)
    # sa = f'{os.sep}images{os.sep}'#  /images/
    train_dict = dict()
    val_dict = dict()
    test_dict = dict()
    img_nbr_dict = dict()

    for img_dir in img_dir_list:
        path_parts_reversed = Path(img_dir).resolve().parts[::-1]
        # если среди каталогов есть 'train'
        if 'train' in path_parts_reversed:
            # добавляем в соотвествующий словарь максимальный уровень
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

        img_nbr_dict[img_dir] = len(
            [f_name for f_name in os.listdir(img_dir)
             if os.path.splitext(f_name)[1] in yolov5_img_video_formats]
            # [f_name for f_name in img_path_list
            # if os.path.split(f_name)[0]==img_dir]
        )

    if train_dict:
        yaml_dict['train'] = list(train_dict.keys())[
            list(train_dict.values()).index(max(train_dict.values()))]
        if yaml_dict['train'] in val_dict.keys():
            del val_dict[yaml_dict['train']]
        if yaml_dict['train'] in test_dict.keys():
            del test_dict[yaml_dict['train']]

        # если train не найден, то остальное ('val', 'test') игнорируем
        if val_dict:
            yaml_dict['val'] = list(val_dict.keys())[
                list(val_dict.values()).index(max(val_dict.values()))]
            if yaml_dict['val'] in test_dict.keys():
                del test_dict[yaml_dict['val']]

        if test_dict:
            yaml_dict['test'] = list(test_dict.keys())[
                list(test_dict.values()).index(max(test_dict.values()))]

    # если не находим по названиям - считаем картинки и в train назначаем max
    if not yaml_dict.get('train', []):
        if img_nbr_dict:
            yaml_dict['train'] = list(img_nbr_dict.keys())[
                list(img_nbr_dict.values()).index(max(img_nbr_dict.values()))]
        else:
            print('не найдены каталоги с изображениями')
            return False

    if not yaml_dict.get('val', []):
        if len(img_nbr_dict) > 1:
            sorted_dir_size = sorted(img_nbr_dict.values())
            val_size = sorted_dir_size[-2]  # -1 - макс - оставлен для train
            yaml_dict['val'] = list(img_nbr_dict.keys())[
                list(img_nbr_dict.values()).index(val_size)]
        else:
            yaml_dict['val'] = yaml_dict['train']
            print(
                f'так как найден только один каталог с изображенями в yaml ключ "val" будет равен "train" и равен {yaml_dict["val"]}')

            # если нет катлога с именем test, по размеру не будем его назначать

    # И как только я это все сделал - попал в ситуацию, когда каталог для валидации назывался test
    # короче это все нужно бросить я думаю
    '''        
    if yaml_dict.get('test',[]):
        #test назначим опционально, если есть свободные каталоги
        if len(img_nbr_dict)>2: 
            sorted_dir_size = sorted(img_nbr_dict.values())
            val_size = sorted_dir_size[-3] #-1, -2 - оставлены для train, val
            yaml_dict['val'] = list(img_nbr_dict.keys())[
                                    list(img_nbr_dict.values()).index(val_size)]
    '''
    # если сюда добрались, значит есть yaml_dict['train'] и yaml_dict['val']

    yaml_dict['names'] = list(class_dict.values())
    yaml_dict['nc'] = len(yaml_dict['names'])
    if yaml_path_param: yaml_dict['path'] = yaml_path_param

    # print('yaml_dict', yaml_dict)

    if yaml_f_name_path:
        if os.path.split(yaml_f_name_path)[0]:
            dest_yaml_path = yaml_f_name_path
        else:
            # можно бы проверить, но кажется не такая уж и ценность
            dest_yaml_path = os.path.join(data_dir, yaml_f_name_path)
        save_data_yaml(yaml_dict, dest_yaml_path, f_overwrite=f_overwrite)
    else:
        print('Путь для записи yaml не задан. Файл не сохранен')
        dest_yaml_path = None

    return dest_yaml_path


def create_yolov5_dataset(data_dir,
                          format_key,
                          yaml_f_name='dataset.yaml',
                          f_overwrite=False):
    # брак

    def is_parent(parent, child):
        par_path = Path(os.path.abspath(parent)).resolve().parts[1:]
        ch_path = Path(os.path.abspath(child)).resolve().parts[1:]
        return par_path == ch_path[:len(par_path)]

    loc_templ_dict = dict(zip(['yolov5', 'coco', 'pascal_voc'],
                              ['*.txt', '*.json', '*.xml']))

    print(f'Создаются аннотации в формате yolov5 для каталога {data_dir}')

    img_path_list, img_dir_list, lab_dir_list, ann_path_list = get_yolov5_images_glob(
        data_dir)  # Not Implemented, f_change_dir_structure=False)

    # img_path_list первый список файлов изображений не должен быть пустым и файлы существуют
    # но он нам особо не нужен, дальше только аннотации смотрим
    # img_dir_list второй список каталогов с изображениями не должен быть пустым и каталоги существуют
    # примерно так img_dir_list =  ['/content/images/train', '/content/images/valid'],
    # а вот каталоги третьего списка для меток может и существовать и не существовать
    # примерно так lab_dir_list = ['/content/labels/train', '/content/labels/valid'],

    # сейчас нужно для каждого каталога из img_dir_list найти файл с аннотациями
    # конвертнуть их и сохранить результат в соотвествующую папку из lab_dir_list

    # здесь у нас могут быть два варианта coco или pascal_voc
    # и там и там аннотации содержатся в папке с изображениями

    class_dict_list = []
    for ind, img_dir in enumerate(img_dir_list):
        # так как каталоги предобработаны рекурсивный поиск сейчас не нужен
        '''
        probably_ann_list = glob.glob(os.path.join(img_dir,
                                                #'*.json'),
                                                loc_templ_dict[format_key]),
                                                #ann_templ_dict[f_key]),
                                                recursive=False)

        '''
        probably_ann_list = [f_path for f_path in ann_path_list if
                             os.path.splitext(f_path)[1] == os.path.splitext(
                                 loc_templ_dict[format_key])[1]]
        probably_ann_list = [f_path for f_path in probably_ann_list if
                             is_parent(img_dir, f_path)]

        if probably_ann_list:
            if format_key == 'coco':
                # вообще файлов с аннотациями coco может быть и много, возьмем первый пока
                ann_file = probably_ann_list[0]
                class_dict_list.append(convert_2_yolov5_ann(ann_file,
                                                            lab_dir_list[ind],
                                                            input_format=format_key))
            elif format_key == 'pascal_voc':
                ann_dir_list = sorted(set(
                    [os.path.split(f_path)[0] for f_path in probably_ann_list]))
                # аналогично, возьмем первый
                ann_dir = ann_dir_list[0]
                class_dict_list.append(convert_2_yolov5_ann(ann_dir,
                                                            lab_dir_list[ind],
                                                            input_format=format_key))
        else:
            print(f'В каталоге {img_dir} не найдены аннотации {format_key}')

    # у нас было несколько каталогов, берем самый большой из справочника классов
    # не безопасно, в идеале нужно слить все справочники
    cl_dict_len_list = [len(cl_dict) for cl_dict in class_dict_list if cl_dict]
    if not cl_dict_len_list:
        print(f'В каталогах {img_dir_list} не найдены классы объектов {format_key}')
        return False
    max_cl_dict_len = max(cl_dict_len_list)
    class_dict = class_dict_list[cl_dict_len_list.index(max_cl_dict_len)]

    return create_yolov5_yaml(data_dir,
                              img_path_list, img_dir_list, lab_dir_list,
                              # probably_ann_list,
                              dest_yaml_path=data_dir,
                              class_dict=class_dict,
                              yaml_f_name_path='dataset.yaml',
                              yaml_path_param=None,  # ключ path
                              f_overwrite=f_overwrite)


def check_yolov5_yaml(yaml_path, base_dir=None, force_base_dir=False):
    """
    :param yaml_path: путь к yaml описанию
    :param base_dir: путь для замены переменной path
    :param force_base_dir: если True - переменная path из yaml_path будет проигнорирована
     на случай перемещеия данных
    :return: True
    NOTE: base_dir будет работать только если в параметрах train, val, test
     - неабсолютные пути
     если base_dir == None, вместо base_path будет попытка взять
     os.path.split(yaml_path)[0], если base_dir не нужен - нужно передать ''

    """

    def get_abs_path(dict_dir):
        if not (os.path.isabs(dict_dir)):
            # dict_dir = os.path.join(base_dir, dict_dir)
            dict_dir = os.path.abspath(os.path.join(base_dir, dict_dir))
        return str(dict_dir)  # чтобы прошла проверка путь должен os.sep заканчиваться

    def check_yolo_img_lab_dir(dict_key):
        # img_dir_for_check = get_abs_path(Path(yaml_dict.get(dict_key, '')))
        img_dir_for_check = get_abs_path(yaml_dict.get(dict_key, ''))
        if not os.path.isdir(img_dir_for_check):
            print(f'Каталог {img_dir_for_check} не найден')
            return ''
        img_f_names = set([os.path.splitext(f_name)[0]
                           for f_name in os.listdir(img_dir_for_check)
                           if os.path.splitext(f_name)[1] in yolov5_img_video_formats])
        if len(img_f_names) < 1:
            print(f'В каталоге {img_dir_for_check} не найдены файлы с изображениями')
            return ''

        # чтобы получился путь к меткам, и последний каталог должен быть строго /images/
        # просто /images без слэша в конце будет ошибка. Если не последний - все равно,

        # на случай если последний каталог == /images нужно добавить сепаратор
        img_dir_for_check = str(img_dir_for_check) + os.sep
        lab_dir_for_check = sb.join(img_dir_for_check.rsplit(sa, 1))

        if not os.path.isdir(lab_dir_for_check):
            print(f'Каталог меток {lab_dir_for_check} не найден')
            return ''

        lab_f_names = set([os.path.splitext(f_name)[0]
                           for f_name in os.listdir(lab_dir_for_check)
                           if os.path.splitext(f_name)[1] == '.txt'])

        if len(img_f_names & lab_f_names) < 1:
            print(f'В каталоге {img_dir_for_check} не найдены подходящие аннотации')
            return ''

        return img_dir_for_check

    result = (dict(), None)

    yaml_dict = get_data_dict(yaml_path)
    if not isinstance(yaml_dict, (dict)):
        print(f'По адресу {yaml_path} не найден словарь с описанием')
        return result

    if base_dir == None:
        base_dir = os.path.split(yaml_path)[0]
    if not force_base_dir:
        base_dir = yaml_dict.get('path', base_dir)
    # base_dir = Path(base_dir)

    train_dir = check_yolo_img_lab_dir('train')
    if not train_dir:
        return result
    val_dir = check_yolo_img_lab_dir('val')

    if not val_dir:
        return result

    if val_dir == train_dir:
        print(f'Kлючи "train" и "val" ссылаются на один каталог')

    names = yaml_dict.get('names', [])
    if len(names) < 1:
        print(f'Не найдены наименования классов')
        return result

    nc = int(yaml_dict.get('nc', 0))

    if nc < 1:
        print(f'Не указано количество классов')
        return result
    return yaml_dict, base_dir


def get_actual_yolov5_yaml(yaml_path,
                           dest_yaml_path=None):
    """
    # функция должна проверить yaml в каталоге
    # и не совсем понятно что должна сделать еще, пока оставим

    :param dataset_dir_yaml_path: путь к датасету или yaml описанию
    :param dest_yaml_path: путь для нового описания

    :return: путь к актуальному yaml, старому или созданному
    """

