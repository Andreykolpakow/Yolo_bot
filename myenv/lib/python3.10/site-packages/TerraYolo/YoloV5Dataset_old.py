#import os

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


# sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'# /images/, /labels/ substrings
# from TerraYoloTest.dataset_utils.ann_convertors import check_yolov5_yaml, sa,sb
from TerraYoloTest.dataset_utils.ann_convertors import sa, sb, yolov5_img_video_formats
from TerraYoloTest.TerraYolo import save_data_yaml, get_data_dict


# from TerraYoloTest.utils.general import fast_scandir


def get_yolo_lab_dir(img_dir):
    return sb.join((str(pathlib.Path(img_dir)) + os.sep).rsplit(sa, 1))


class YoloV5Dataset:
    def __init__(self,
                 yaml_path_dir=None,
                 train_img_dir=None, val_img_dir=None, test_img_dir=None,
                 # data_root_path=None,
                 # cl_labels=None,
                 names=None,

                 ):

        if not os.path.exists(yaml_path_dir):
            print(f'файл/каталог {yaml_path_dir} с данными не найден')
            return None

        self.data_keys = ['train', 'val', 'test']
        self.data_dict = dict()
        for k in self.data_keys:
            self.data_dict[k] = dict()

        # на этом шаге ищем каталоги с изображениями
        if os.path.isfile(yaml_path_dir):
            # разница в том, что в parse_yaml сразу проверяется соответствие меток
            self.parse_yaml(yaml_path)
            # те если есть yaml то есть и датасает yolo
        elif os.path.isdir(yaml_path_dir):
            # а если yaml нет, то возможно yolo метки нужно будет еще только сделать
            # if names:
            # else print()
            self.explore_base_dir(base_dir=yaml_path_dir)
            # else:
            # self.explore_dir_list(train_img_dir, val_img_dir, test_img_dir)

        if self.data_dict:
            self.train_dict = self.data_dict['train']
            self.val_dict = self.data_dict.get('val', dict())
            self.test_dict = self.data_dict.get('test', dict())
        else:
            print(f'В файл/каталог {yaml_path_dir} с данными не найден')
            return None

        # на этом шаге ищем каталоги с аннотациями

        # анализ изображений и аннотаций если чего-то не хватает - предложить поискать еще

        # print('self.train_dict', self.train_dict, self.data_dict['train'])
        # yaml_path  в приоритете, если не найдется yaml_dict - пробуем data_root_path
        # else:
        # pass
        # if self.parse_root_path(data_root_path)

        # в этом месте должен быть self.train - из yaml или структуры каталогов
        # нет train - нет датасета

        if data_root_path:
            self.path = data_root_path
        # self.classes
        # self.nc
        pass

    def parse_yaml(self, yaml_path):
        yaml_dict, base_dir = check_yolov5_yaml(yaml_path)
        if yaml_dict:
            # смысл такой, что path может и не быть, а base_dir можно вычислить или задать
            self.base_dir = base_dir
            self.path = yaml_dict.get('path', '')
            # если пройдена check_yolov5_yaml то 'train' и 'val' обязаны быть
            # и соответствующие каталоги с метками тоже и содержать метки
            self.data_dict['train']['img_dir'] = yaml_dict['train']
            self.data_dict['val']['img_dir'] = yaml_dict['val']
            if yaml_dict.get('test'):
                self.data_dict['test']['img_dir'] = yaml_dict['test']

            self.names = yaml_dict['names']
            self.nc = int(yaml_dict['nc'])
            return True
        return False

    def explore_base_dir(self, base_dir, name_priority=True):
        # вообще если пришел каталог, то надо бы в первую очередь искать yaml
        # потому что без него нет информации по классам
        yaml_list = [f_name for f_name in os.listdir(base_dir) if
                     os.path.splitext(f_name) == '.yaml']
        print(yaml_list)

        img_dict = explore_img_dir(base_dir)
        if not img_dict:
            return None

        train_dict, val_dict, test_dict = img_dir_name_analyze(
            img_dir_dict=img_dict)

    def explore_dir_list(self, train_img_dir, val_img_dir, test_img_dir):
        # NotImplemented
        pass

    def get_img_info(self):
        # тут надо загрузить информацию об изображениях и подгружать по мере необходимости новую
        # NotImplemented
        pass

    def train_test_split(self):
        # NotImplemented
        pass

    def save(self, dest_path):
        # сохранить датасет
        # NotImplemented
        pass

    def load(self, source_path):
        # загрузить датасет
        # NotImplemented
        pass

    def save_yaml(self):
        # сохранить yaml
        # NotImplemented
        pass

    def add_exp(self):
        # добавить в датасет результаты эксперимента
        # NotImplemented
        pass


#yolo_v5_dataset = YoloV5Dataset(yaml_path_dir=r'C:/WORK/NU/YOLO/data/10K/')
'''
# yolo_v5_dataset = YoloV5Dataset(yaml_path=r'C:\WORK\NU\YOLO\data\10K\data_path.yaml')
yolo_v5_dataset = YoloV5Dataset(yaml_path=r'C:/WORK/NU/YOLO/data/10K/data_path.yaml')
dir(yolo_v5_dataset)

print('get_yolo_lab_dir', get_yolo_lab_dir(r'C:\WORK\NU\YOLO\data\10K\images\\'))
print(get_yolo_lab_dir(r'C:\WORK\NU\YOLO\data\10K\images'))
print(get_yolo_lab_dir(r'C:\WORK\NU\YOLO\data\images\10K'))
print(get_yolo_lab_dir(r'C:\WORK\NU\YOLO\data\images\10K\\'))
'''
