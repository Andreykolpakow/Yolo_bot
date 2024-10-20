import os
import random
import shutil
from math import ceil

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
from PIL import Image

#print(f'{__file__}, {__name__}, {__package__}')
""" Набор функций для манипулирования  датасетами yolov5"""

def get_lab_classes(lab_list):

    #Функция принимает список меток yolov5 и возвращает список первых значений как список классов

    return [int(line.split(' ')[0]) for line in lab_list]


def get_file_labels(lab_path):
    """
    Функция парсит координаты из текстового файла

    :param lab_path: путь к файлу аннотаций
    :return:
    lab_list:list - список списков, где каждый элемент списка соответствует метке,
    первое значение  каждого вложенного списка - классу объекта
    следующие значения  - координатам рамок
    lab_classes_list: - список классов объектов приведенных к типу int,
    соответствующих каждой рамке

    NOTE: метки не проверяются, в результат попадут списки соответствующие всем строкам
    NOTE: классы и координаты в lab_list типа строка, в классы в lab_classes_list - int
    """

    lab_list, lab_classes_list = [], []
    if os.path.isfile(lab_path):
        with open(lab_path, 'r') as ann_file:
            lab_list = ann_file.read()
            lab_list = lab_list.splitlines()
            lab_list = [lab for lab in lab_list]
            lab_classes_list = get_lab_classes(lab_list)
    return lab_list, lab_classes_list


def get_filtered_label(lab_path, class_filter: list):
    """

    :param lab_path: путь к файлу с аннотацией
    :param class_filter: массив вида [class_id_1_int, class_id_2_int, ...]
    :return: список списков где элементы вложенного списка (типа str) соответствуют
    координатам bbox. Первый - классу объекта, остальные - координатам
    NOTE: координаты  НЕ проверяются, в результат попадут списки соответствующие
    всем строкам, удовлетворяющим фильтру
    """
    res = []
    if os.path.isfile(lab_path):
        lab_list, lab_classes_list = get_file_labels(lab_path)
        mask = np.in1d(lab_classes_list, class_filter)
        if np.any(mask) == True:
            res = list(np.array(lab_list)[mask])
    return res




def xywhn2xyxy_uno(x, w=640, h=640):

    """
    Преобразование координат bbox для вывода рамок объекта
    переработано из утилит yolov5
    # from general.py import xywhn2xyxy
    в yolo все функции выполнены под матричные и тензорные операции

    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)

    в тч для наглядности переделано под едничные значения _uno
    """


    y = np.copy(x)
    y[0] = w * (x[0] - x[2] / 2)  # + padw  # top left x
    y[1] = h * (x[1] - x[3] / 2)  # + padh  # top left y
    y[2] = w * (x[0] + x[2] / 2)  # + padw  # bottom right x
    y[3] = h * (x[1] + x[3] / 2)  # + padh  # bottom right y
    return y


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


def transform_label(trans_dict, label_dir, new_label_dir=None, class_filter=None):

    """
    Экспериментальная заготовка модуля трансформации координат


    :param trans_dict:
    :param label_dir:
    :param new_label_dir:
    :param class_filter:
    :return:
    """
    if not (new_label_dir):
        new_label_dir = label_dir
        #print(new_label_dir)

    if os.path.isdir(new_label_dir) == False:
        os.mkdir(new_label_dir)

    # xywhn метки ожидаются
    # trans_dict.keys = 'stretch'
    def stretch(class_xywh_list, param):
        """
        Растяжение (сжатие) рамки, умножает предпоследнюю и последнюю координату рамки
        на множитель param
        в перспективе множители могли бы быть разные по ширине и высоте
        а также адаптивные к размеру метки и изображения

        :param class_xywh_list:
        :param param:
        :return:
        """
        # растягиваем рамку на заданый  множитель param
        # print(params)
        return class_xywh_list[:-2] + [str(float(class_xywh_list[-2]) * param),
                                       str(float(class_xywh_list[-1]) * param)]

    metaop_dict = dict()
    metaop_dict['stretch'] = stretch

    for k, param in trans_dict.items():
        for f_name in os.listdir(label_dir):
            lab_list, _ = get_file_labels(os.path.join(label_dir, f_name))
            new_lab_list = []
            for label in lab_list:
                if class_filter:
                    # print(f_name, lab_list, class_filter)
                    # if (class_filter): print('class_filter')
                    # break
                    ann_list = label.split()
                    if int(ann_list[0]) in class_filter:
                        new_ann = metaop_dict[k](ann_list, param)
                        new_lab_list.append(' '.join(new_ann))
                else:
                    new_lab_list.append(label)
            if len(new_lab_list) > 0:
                #отладочно if len(new_lab_list) > 1:
                    #print(f_name, len(new_lab_list))
                with open(os.path.join(new_label_dir, f_name), 'w') as f:
                    # print(new_lab_list)
                    f.write('\n'.join(new_lab_list))


def copy_labels(source_lab_dir, dest_lab_dir,
                class_filter=None,
                filter_label=True
                ):
    """
    Функция копирует файлы с аннотациями, содержащими классы из class_filter
    из source_lab_dir в dest_lab_dir

    :param source_lab_dir:
    :param dest_lab_dir:
    :param class_filter:
    :param filter_label: если True - из файлов аннотаций будут также удалены метки
    классов, не входящих в class_filter
    :return:

    NOTE: если class_filter не задан будут скопированы все файлы

    """
    if (os.path.isdir(source_lab_dir) == False) or (
            len(os.listdir(source_lab_dir)) == 0):
        print(source_lab_dir, 'не найден или пуст каталог с метками')
        return

    if os.path.isdir(dest_lab_dir) == False:
        # будет ошибка если нет каталого верхнего уровня
        os.mkdir(dest_lab_dir)

    if class_filter == None:
        # тут и выше нужен фильтр по txt
        #shutil.copytree(source_lab_dir, dest_lab_dir)
        for f_name in os.listdir(source_lab_dir):
            if os.path.isfile(os.path.join(source_lab_dir, f_name)):
                shutil.copy(os.path.join(source_lab_dir, f_name),
                            os.path.join(dest_lab_dir, f_name))
    else:
        for f_name in os.listdir(source_lab_dir):
            source_path = os.path.join(source_lab_dir, f_name)
            lab_filtered_list = get_filtered_label(source_path, class_filter)
            if len(lab_filtered_list) > 0:
                '''
                for lab in lab_filtered_list:
                    if len(lab.split())<5: 
                        print('source_path', source_path, len() )
                        break
                '''
                # !cat {source_path}
                # print('\n**************')
                # print(f_name, lab_filtered_list)
                if filter_label == True:
                    # print(dest_lab_path+os.sep+f_name, lab_filtered_list)
                    with open(os.path.join(dest_lab_dir, f_name), 'w') as lab_filtered_f:
                        # print('\n'.join(lab_filtered_list))
                        # lab_filtered_f.write('\n'.join(lab_filtered_list))
                        lab_filtered_f.write('\n'.join(lab_filtered_list))
                else:
                    shutil.copyfile(source_path, dest_lab_dir+f_name)
                # break
            pass


def show_images(img_list, box_list,
                ncols=4,
                show_box=False,
                show_label=0,
                name_list=(),
                img_size=(4, 3),
                show_conf=False,
                round_conf=3
                ):
    """
    Функция выводит изображения из списка img_list с рамками из box_list
    :param img_list: список путей к файлам изображений
    :param box_list: список меток обектов, соответствующих файлам изображений
    :param ncols:
    :param show_box: если True - вывести bbox
    :param show_label: 0 - не выводить метки, 1- выводить номер класса, 2 - имя класса
    :param img_size:
    :return:
    """

    if show_label == 2:
        if len(name_list) < 1:
            print(f"Не заданы имена классов, выводим номера")
            show_label = 1
    nrows = ceil(len(img_list) / ncols)
    # print(img_list, box_list, nrows, ncols)
    figsize = np.array([img_size[0] * ncols, img_size[1] * nrows])
    fig, axs = plt.subplots(nrows, ncols=ncols, figsize=figsize)
    axs = axs.flat
    for ind, f_name in enumerate(img_list):
        ax = axs[ind]
        # print(f_name)
        with Image.open(f_name) as img:
            ax.imshow(img)
            if show_box == True:
                for box in box_list[ind]:
                    box_data = box.split(' ')
                    cl_id = int(box_data[0])
                    box_xywhn = [float(coord) for coord in box_data[1:]]
                    box_xyxy = xywhn2xyxy_uno(box_xywhn,
                                              w=img.width,
                                              h=img.height)
                    boxcolor=[color/255 for color in yolo_colors(cl_id)]
                    ax.add_patch(
                        patches.Rectangle((box_xyxy[0:2]), box_xyxy[2] - box_xyxy[0],
                                          box_xyxy[3] - box_xyxy[1],
                                          linewidth=1,
                                          edgecolor=boxcolor,
                                          facecolor='none'))

                    box_txt=''
                    if show_label > 0:
                        # cl_id = int(box_data[0]) и наоборот
                        if show_label < 2:
                            box_txt = box_data[0]
                        else:
                            if cl_id > len(name_list):
                                box_txt = "UNKNOWN LABEL " + box_data[0]
                            else:
                                box_txt = name_list[cl_id]
                        # left_bottom = (box_xyxy[0], box_xyxy[3])
                    if show_conf:
                        try:
                            box_txt += f' conf:{round(float(box_data[5]),round_conf)}'
                        except:
                            print(f'Для {f_name} порог conf нет данных')
                    if box_txt:
                        ax.text(box_xyxy[0]+1, box_xyxy[3]-2, box_txt, color=boxcolor)
    plt.show()
    return


def count_classes_in_lab_dir(lab_dir, show=False, figsize=(10,5), names=None):
    # заодно посмотрим баланс классов

    """
    Подсчет классов объектов в файлах аннотаций из каталога lab_dir
    :param lab_dir: каталог с описаниями
    :param show: флаг диаграммы
    :param figsize: размер диаграммы, если не задан, то по умолчанию
    :param names: имена классов если не задан  - то номера классов
    :return:
    """
    print('lab_dir', lab_dir)
    if os.path.isdir(lab_dir) != True:
        print('Не найден каталог с изображениями')
        return

    box_list = []
    for f_name in os.listdir(lab_dir):
        with open(os.path.join(lab_dir, f_name), 'r') as f:
            boxes = f.readlines()
            try:
                box_list += [int(box.split(' ')[0]) for box in boxes]
            except:
                print(f'{os.path.join(lab_dir, f_name)} не найдены {boxes}')
                return
    value, counts = np.unique(box_list, return_counts=True)

    if show:
        if figsize:
            plt.figure(figsize=figsize)
        if not names:
            names = value
        plt.bar(names, counts)

    return value, counts, len(os.listdir(lab_dir)), lab_dir  # , counts/np.sum(counts)

def cut_labels(lab_dir):
    """
    Проверяет все файлы из каталога lab_dir и оставляет в каждой строке не больше
    пяти значений (четырех пробелов): подразумевается, что в каждой строке останется
    class + xywhn

    :param lab_dir:
    :return:
    NOTE: проверка bbox не производится, например bbox с недостаточным количеством
    координат останутся в файле
    функция появилась под конкретную разметку, скорей всего не нужна
    """

    for f_name in os.listdir(lab_dir):
        with open(os.path.join(lab_dir, f_name), 'r') as f:
            box_list = f.read().splitlines()
            lab_cut = [' '.join(box.split(' ')[:5]) for box in box_list]
        with open(os.path.join(lab_dir, f_name), 'w') as f:
            f.write('\n'.join(lab_cut))
    return


def check_lab_by_name(new_lab_dir, exist_lab_dir):
    """
    Функция по именам файлов без расширений проверяет наличие изображения для
    соответствующего файла аннотации

    :param new_lab_dir: путь к каталогу новых аннотаций предполагается, что эти метки
     будут добавляться в существующий каталог)
    :param exist_lab_dir: путь к каталогу обновляемых меток
    :return: возвращает мнодество общих имкен файлов - если пустое - старые аннотации
    однозначно не будут изменены при добавлении
    """

    # можно также под список директорий любой длины сделать только не ясно нужно ли
    new_lab_f_name_set = set([name for name in os.listdir(new_lab_dir)
                              if name.endswith('txt')])
    exist_lab_f_name_set = set([name for name in os.listdir(exist_lab_dir)
                                if name.endswith('txt')])

    return new_lab_f_name_set & exist_lab_f_name_set


def check_img_for_lab(lab_dir, img_dir):
    """
    Функция по именам файлов без расширений проверяет наличие изображения для
    соответствующего файла аннотации
    """

    lab_f_name_set = set([name[:-4] for name in os.listdir(lab_dir)
                          if name.endswith('txt')])
    img_f_name_set = set([name[:-4] for name in os.listdir(img_dir)
                          if name.endswith('jpg')])
    return lab_f_name_set & img_f_name_set


def add_img_by_lab(lab_dir, img_source_dir, img_dest_dir):
    lab_f_name_set = set([name[:-4] for name in os.listdir(lab_dir)
                          if name.endswith('txt')])

    # можно сделать словарем имя без расширения: полное имя
    img_f_name_set = set([name[:-4] for name in os.listdir(img_source_dir)
                          if name.endswith('jpg')])
    img_f_name_set = img_f_name_set & lab_f_name_set
    print(f'найдено {len(img_f_name_set)} для {len(lab_f_name_set)} файлов аннотаций')

    for f_name in img_f_name_set:
        # файлы *.JPG *.JPg и тп не скопирует
        shutil.copy(os.path.join(img_source_dir, f_name, '.jpg') , img_dest_dir)


def show_class_images(img_dir,
                      show_box=False,
                      lab_dir=None,
                      show_label=0,
                      name_list=(),
                      img_size=(4, 3),
                      class_filter=None, #если не задать - ограничит без сообщения
                      ncols=4,
                      # subdir_list=['train'],
                      img_nbr=8,
                      n_part=1,  # число частей, на которые разобъем данные
                      # номер части питон (0..n_parts-1) если n_parts определено а
                      # parts_nbr - нет: покажем случайную, parts_nbr>n_parts - последнюю
                      part_nbr=None,
                      show_conf=False
                      # не сделал, задумано как интерактивный вывод всех изображений
                      # interact=False
                      ):
    """
    Функция выводит на экран изображения в несколько колонок
    с фильтром по классам class_filter
    :param img_dir: путь к изображениям
    :param show_box: флаг вывода bbox
    :param lab_dir: путь к меткам, нужен независимо от значения show_box
    :param show_label: способ вывода меток - 0 - не показывать, 1 - номер, 2 -имя класса
        (из name_list)
    :param name_list: имена классов, если не заполнены, то будут номера
    :param img_size: размер одного изображения
    :param class_filter: фильтр для изображений, если не задан то будет отфильтрован (0)
    :param ncols: число колонок для вывода
    :param img_nbr: число изображений для вывода, если  img_nbr==0 - выведутся все
        изображения, подходящие фильтру классов
    :param n_part: если не задан параметр img_nbr, то весь список разобъется на
    n_part частей
    :param part_nbr: номер части для вывода на экран из списка, разделенного на n_part
    частей
    :return:
    NOTE: параметры n_part и part_nbr применяются для последовательного вывода длинных
    списков изображений по частям

    """

    if class_filter == None:
        print(f'class_filter не задан, устанавливаем class_filter=[0]')
        class_filter = [0]

    print(f"""
            Пробуем вывести {str(img_nbr)+' изображений' if img_nbr>0 
                                        else 'все изображения'}  из каталога {img_dir}, 
            {'с рамками ' if show_box else 'без рамок'},
            с фильтром классов {class_filter}, 
        """)
    if os.path.isdir(img_dir) != True:
        print('Не найден каталог с изображениями')
        return

    if show_box == True:
        #print('lab_dir', lab_dir)
        #print(os.path.isdir(lab_dir))
        if os.path.isdir(lab_dir) != True:
            # Если не задан попытаемся найти относительно изображений
            #new_lab_dir = os.path.abspath(img_dir + os.sep + '..' + os.sep + 'labels'
                                          #+ os.sep)
            new_lab_dir = os.path.join(img_dir, '..', 'labels')
            if lab_dir:
                print(
                    f'Не найден каталог {lab_dir}, пробуем найти метки в каталоге'
                    f' {new_lab_dir}')
            if os.path.isdir(new_lab_dir) == False:
                print('каталог с метками не найден')
                show_box = False
                return
            else:
                lab_dir = new_lab_dir

    if not os.path.isdir(lab_dir):
        print('Не найден каталог с аннотациями - нет данных для фильтра класса')
        return


    #class_filter_arr = np.array(class_filter)
    # def get_other_path():
    img_filtered, box_filtered = [], []
    # for subdir in subdir_list:

    # попробуем набрать в цикле img_nbr картинок для вывода
    for f_name in os.listdir(lab_dir):
        # пока пропускаем случай, когда img_nbr=-1
        with open(os.path.join(lab_dir, f_name), 'r') as ann_file:
            lab_list = ann_file.read().splitlines()
            lab_classes_list = np.array(get_lab_classes(lab_list))
            mask = np.in1d(lab_classes_list, class_filter)
            if np.any(mask) == True:
                # print('len(mask)',len(mask), mask, f_name)
                #img_filtered.append(img_dir + f_name[:-3] + 'jpg')
                img_filtered.append(os.path.join(img_dir, f_name[:-3] + 'jpg'))
                box_filtered.append(list(np.array(lab_list)[mask]))
        # print(img_nbr==False, f_name, len(img_filtered), box_filtered)
        if img_nbr > 0:
            if (len(img_filtered) > 0) & (len(img_filtered) % img_nbr == 0):
                show_images(img_filtered, box_filtered,
                            ncols=ncols,
                            show_box=show_box,
                            show_label=show_label,
                            name_list=name_list,
                            img_size=img_size,
                            show_conf=show_conf
                            )
                img_filtered, box_filtered = [], []
                # набрали img_nbr картинок, вывели их, обнулили списки и прервались
                break

    # если img_nbr==0 тогда нашли все картинки и выбираем только часть из них
    # или если img_nbr>0 но нужного количества rfhnbyjr не нашли - выведем, те, что нашли

    if len(img_filtered) > 0:
        print(len(img_filtered), ' images filtered from classes ', class_filter)
        start_idx, end_idx = 0, None  # покажем все если что-то пойдет не так
        if img_nbr <= 0:
            #print(img_nbr, n_part)
            if n_part > 0:
                part_len = len(img_filtered) / n_part
                if (part_nbr == None) or (part_nbr < 0):  # если n_part задано, а part_nbr не задано или меньше 0 - покажем случайную часть из n_part
                    part_nbr = random.randint(0, n_part - 1)
                # если part_nbr>n_part - покажем последнюю часть из n_part
                elif part_nbr >= n_part:
                    part_nbr = n_part - 1

                start_idx = round(part_len * part_nbr)
                end_idx = round(start_idx + part_len)
        print(f'show images from {start_idx} to {end_idx}')
        show_images(img_filtered[start_idx:end_idx],
                    box_filtered[start_idx:end_idx],
                    ncols=ncols,
                    show_box=show_box,
                    show_label=show_label,
                    name_list=name_list,
                    img_size=img_size,
                    show_conf=show_conf
                    )


# из yolov5 репозитория взято
class Colors:
# Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

yolo_colors=Colors()

def show_list_images(img_f_path_list, ncols=2, img_size=(5,5)):

    """
    Вывод изображений (только изображений, без меток) на экран
    :param img_f_path_list: путь к изображениям
    :param ncols: количество колонок дл вывода
    :param img_size: размер одного изображения
    :return:
    """

    nrows = ceil(len(img_f_path_list) / ncols)
    figsize = np.array([img_size[0] * ncols, img_size[1] * nrows])
    fig, axs = plt.subplots(nrows, ncols=ncols, figsize=figsize)
    if len(img_f_path_list) > 1:
        axs = axs.flat
        for ind, f_name in enumerate(img_f_path_list):
            ax = axs[ind]
            with Image.open(f_name) as img:
                ax.imshow(img)
    else:
        with Image.open(img_f_path_list[0]) as img:
            axs.imshow(img)

    plt.show()


def get_img_sizes(img_path, sample_size=0):
    """
    Функция собирает информацию о размерах изображений
    :param img_path: путь к файлам с изображениями
    :param sample_size: количество, если >1 - то количество изображений для вывода
     если sample_size от 0 до 1 - доля от общей выборки
     sample_size = 1.  приравнивается к 1, если нужна вся выборка - можно указать 0
    :return: возвращает массив numpy с именами файлов и двумерный массив с размерами
    """

    if not os.path.isdir(img_path):
        print(f'Путь к файлам изображений не найден: {img_path}!')
        return
    img_f_list = sorted([f_name for f_name in os.listdir(img_path)
                        if os.path.splitext(f_name)[1].lower() == '.jpg'])
    if not len(img_f_list)>0:
        print(f'По адресу {img_path} файлы изображений не найдены!')
        return

    if sample_size >= 1:
        img_f_list = random.sample(img_f_list, min(round(sample_size), len(img_f_list)))
    elif sample_size > 0:
        img_f_list = random.sample(img_f_list, round(sample_size*len(img_f_list)))

    img_wth_list = []
    img_ht_list = []
    for img_f_name in img_f_list:
        img = Image.open(os.path.join(img_path,img_f_name))
        img_wth_list.append(int(img.width))
        img_ht_list.append(int(img.height))
    res = np.array([img_wth_list, img_ht_list]).T

    return res, np.array(img_f_list)

# TODO убрать отладочные сообщения
