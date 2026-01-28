# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao 
@Time    : 2021/3/24 9:36 上午
@FileName: config.py
@desc: Конфигурация проекта, здесь можно настраивать параметры, жертвуя временем для точности или точностью для времени
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import configparser
import os
import re
import time
from pathlib import Path
try:
    # Для новой версии filesplit (>=4.0.0)
    from filesplit.split import Split
    from filesplit.merge import Merge
    FILESPLIT_NEW_API = True
except ImportError:
    # Для старой версии filesplit (<4.0.0)
    from fsplit.filesplit import Filesplit
    FILESPLIT_NEW_API = False
import paddle
from tools.constant import *

# Версия проекта
VERSION = "2.0.3"

# Базовый каталог проекта
BASE_DIR = str(Path(os.path.abspath(__file__)).parent)

# ×××××××××××××××××××× [НЕ ИЗМЕНЯТЬ] Чтение конфигурационных файлов start ××××××××××××××××××××
# Чтение settings.ini
settings_config = configparser.ConfigParser()
MODE_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.ini')
if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.ini')):
    # Если нет конфигурационного файла, по умолчанию используется русский
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.ini'), mode='w', encoding='utf-8') as f:
        f.write('[DEFAULT]\n')
        f.write('Interface = Русский\n')
        f.write('Language = ru\n')
        f.write('Mode = fast')
settings_config.read(MODE_CONFIG_PATH, encoding='utf-8')

# Чтение языковой конфигурации в interface, например ru.ini
interface_config = configparser.ConfigParser()
INTERFACE_KEY_NAME_MAP = {
    'Русский': 'ru',
    '简体中文': 'ch',
    '繁體中文': 'chinese_cht',
    'English': 'en',
    '한국어': 'ko',
    '日本語': 'japan',
    'Tiếng Việt': 'vi',
    'Español': 'es'
}
interface_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interface',
                              f"{INTERFACE_KEY_NAME_MAP[settings_config['DEFAULT']['Interface']]}.ini")
interface_config.read(interface_file, encoding='utf-8')
# ×××××××××××××××××××× [НЕ ИЗМЕНЯТЬ] Чтение конфигурационных файлов end ××××××××××××××××××××


# ×××××××××××××××××××× [НЕ ИЗМЕНЯТЬ] Проверка корректности пути запуска программы start ××××××××××××××××××××
# Если путь программы содержит китайские символы или пробелы, в процессе работы могут возникать ошибки
# По умолчанию путь считается корректным
IS_LEGAL_PATH = True
# Если путь содержит китайские символы, путь считается некорректным
if re.search(r"[\u4e00-\u9fa5]+", BASE_DIR):
    IS_LEGAL_PATH = False
# Если путь содержит пробелы, путь считается некорректным
if re.search(r"\s", BASE_DIR):
    IS_LEGAL_PATH = False
# Если программа находится в некорректном пути, постоянно выводить предупреждение
while not IS_LEGAL_PATH:
    print(interface_config['Main']['IllegalPathWarning'])
    time.sleep(3)
# ×××××××××××××××××××× [НЕ ИЗМЕНЯТЬ] Проверка корректности пути запуска программы end ××××××××××××××××××××


# ×××××××××××××××××××× [НЕ ИЗМЕНЯТЬ] Определение использования GPU start ××××××××××××××××××××
# Использовать ли GPU (Nvidia)
USE_GPU = False
# Если paddlepaddle скомпилирован с поддержкой GPU
if paddle.is_compiled_with_cuda():
    # Проверить доступность GPU
    if len(paddle.static.cuda_places()) > 0:
        # Если есть GPU, использовать GPU
        USE_GPU = True

# Использовать ли ONNX (DirectML/AMD/Intel)
ONNX_PROVIDERS = []
if USE_GPU == False:
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        for provider in available_providers:
            if provider in [
                "CPUExecutionProvider"
            ]:
                continue
            if provider not in [
                "DmlExecutionProvider",         # DirectML, для Windows GPU
                "ROCMExecutionProvider",        # AMD ROCm
                "MIGraphXExecutionProvider",    # AMD MIGraphX
                # "VitisAIExecutionProvider",   # AMD VitisAI, для RyzenAI & Windows
                "OpenVINOExecutionProvider",    # Intel GPU
                "MetalExecutionProvider",       # Apple macOS
                "CoreMLExecutionProvider",      # Apple macOS
                "CUDAExecutionProvider",        # Nvidia GPU
            ]:
                print(interface_config['Main']['OnnxExectionProviderNotSupportedSkipped'].format(provider))
                continue
            print(interface_config['Main']['OnnxExecutionProviderDetected'].format(provider))
            ONNX_PROVIDERS.append(provider)
    except ModuleNotFoundError as e:
        print(interface_config['Main']['OnnxRuntimeNotInstall'])
if len(ONNX_PROVIDERS) > 0:
    USE_GPU = True
# ×××××××××××××××××××× [НЕ ИЗМЕНЯТЬ] Определение использования GPU end ××××××××××××××××××××


# ×××××××××××××××××××× [НЕ ИЗМЕНЯТЬ] Чтение языка, пути модели, пути словаря start ××××××××××××××××××××
# Установка языка распознавания
REC_CHAR_TYPE = settings_config['DEFAULT']['Language']

# Установка режима распознавания
MODE_TYPE = settings_config['DEFAULT']['Mode']
ACCURATE_MODE_ON = False
if MODE_TYPE == 'accurate':
    ACCURATE_MODE_ON = True
if MODE_TYPE == 'fast':
    ACCURATE_MODE_ON = False
if MODE_TYPE == 'auto':
    if USE_GPU:
        ACCURATE_MODE_ON = True
    else:
        ACCURATE_MODE_ON = False
# Каталог файлов модели
# Версия модели по умолчанию V4
MODEL_VERSION = 'V4'
# Модель детектирования текста
DET_MODEL_BASE = os.path.join(BASE_DIR, 'models')
# Установка модели распознавания текста + словарь
REC_MODEL_BASE = os.path.join(BASE_DIR, 'models')
# Путь к словарю
DICT_PATH = os.path.join(BASE_DIR, 'ppocr', 'utils', 'dict', f'{REC_CHAR_TYPE}_dict.txt')

# Если файл словаря не существует, используем английский словарь по умолчанию
if not os.path.exists(DICT_PATH) and REC_CHAR_TYPE != 'en':
    DICT_PATH = os.path.join(BASE_DIR, 'ppocr', 'utils', 'dict', 'en_dict.txt')
    
# V3, V4 модели по умолчанию shape распознавания изображения 3, 48, 320
REC_IMAGE_SHAPE = '3,48,320'
REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec')
DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_det')

LATIN_LANG = [
    'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
    'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
    'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
    'sw', 'tl', 'tr', 'uz', 'vi', 'latin', 'german', 'french'
]
ARABIC_LANG = ['ar', 'fa', 'ug', 'ur']
CYRILLIC_LANG = [
    'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
    'dar', 'inh', 'che', 'lbe', 'lez', 'tab', 'cyrillic'
]
DEVANAGARI_LANG = [
    'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
    'sa', 'bgc', 'devanagari'
]
OTHER_LANG = [
    'ch', 'japan', 'korean', 'en', 'ta', 'kn', 'te', 'ka',
    'chinese_cht',
]
MULTI_LANG = LATIN_LANG + ARABIC_LANG + CYRILLIC_LANG + DEVANAGARI_LANG + \
             OTHER_LANG

DET_MODEL_FAST_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det_fast')


# Если установлен тип языка распознавания текста, установить соответствующий язык
if REC_CHAR_TYPE in MULTI_LANG:
    # Определение модели детектирования и распознавания текста
    # При использовании быстрого режима, использовать легковесную модель
    if MODE_TYPE == 'fast':
        DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det_fast')
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec_fast')
    # При использовании автоматического режима, определить использование GPU для выбора модели
    elif MODE_TYPE == 'auto':
        # Если используется GPU, использовать большую модель
        if USE_GPU:
            DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det')
            # Для английского режима модель ch распознает лучше, чем fast
            if REC_CHAR_TYPE == 'en':
                REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'ch_rec')
            else:
                REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec')
        else:
            DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det_fast')
            REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec_fast')
    else:
        DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det')
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec')
    # Если в версии по умолчанию (V4) нет большой модели, переключиться на fast модель версии по умолчанию (V4)
    if not os.path.exists(REC_MODEL_PATH):
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec_fast')
    # Если в версии по умолчанию (V4) нет ни большой модели, ни fast модели, использовать большую модель версии V3
    if not os.path.exists(REC_MODEL_PATH):
        MODEL_VERSION = 'V3'
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec')
    # Если в версии V3 нет большой модели, использовать fast модель версии V3
    if not os.path.exists(REC_MODEL_PATH):
        MODEL_VERSION = 'V3'
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec_fast')

    if REC_CHAR_TYPE in LATIN_LANG:
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'latin_rec_fast')
    elif REC_CHAR_TYPE in ARABIC_LANG:
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'arabic_rec_fast')
    elif REC_CHAR_TYPE in CYRILLIC_LANG:
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'cyrillic_rec_fast')
    elif REC_CHAR_TYPE in DEVANAGARI_LANG:
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'devanagari_rec_fast')

    # Определение shape распознавания изображения
    if MODEL_VERSION == 'V2':
        REC_IMAGE_SHAPE = '3,32,320'
    else:
        REC_IMAGE_SHAPE = '3,48,320'

    # Проверить, есть ли полный файл модели распознавания текста в этом пути, если нет, объединить мелкие файлы для создания полного файла
    if 'inference.pdiparams' not in (os.listdir(REC_MODEL_PATH)):
        if FILESPLIT_NEW_API:
            merge = Merge(inputdir=REC_MODEL_PATH, outputdir=REC_MODEL_PATH, outputfilename='merged_file')
            merge.merge()
        else:
            fs = Filesplit()
            fs.merge(input_dir=REC_MODEL_PATH)
    # Проверить, есть ли полный файл модели распознавания текста в этом пути, если нет, объединить мелкие файлы для создания полного файла
    if 'inference.pdiparams' not in (os.listdir(DET_MODEL_PATH)):
        if FILESPLIT_NEW_API:
            merge = Merge(inputdir=DET_MODEL_PATH, outputdir=DET_MODEL_PATH, outputfilename='merged_file')
            merge.merge()
        else:
            fs = Filesplit()
            fs.merge(input_dir=DET_MODEL_PATH)
# ×××××××××××××××××××× [НЕ ИЗМЕНЯТЬ] Чтение языка, пути модели, пути словаря end ××××××××××××××××××××


# --------------------- Измените согласно вашей ситуации start-----------------
# Генерировать ли текстовые субтитры TXT
GENERATE_TXT = False

# Распознавать текст в 6 текстовых областях одновременно на каждом изображении, чем больше видеопамяти GPU, тем больше можно установить это значение
REC_BATCH_NUM = 6
# DB алгоритм распознает сколько изображений в каждом batch, по умолчанию 10
MAX_BATCH_SIZE = 10

# Область появления субтитров по умолчанию - нижняя
DEFAULT_SUBTITLE_AREA = SubtitleArea.UNKNOWN

# Сколько кадров в секунду захватывать для распознавания OCR
EXTRACT_FREQUENCY = 3

# Допустимое отклонение пикселей
PIXEL_TOLERANCE_Y = 50  # Допускается продольное отклонение рамки детектирования на 50 пикселей
PIXEL_TOLERANCE_X = 100  # Допускается горизонтальное отклонение рамки детектирования на 100 пикселей

# Смещение области субтитров
SUBTITLE_AREA_DEVIATION_PIXEL = 50

# Наиболее вероятная область водяного знака
WATERMARK_AREA_NUM = 5

# Порог схожести текста
# Используется для определения, являются ли две строки субтитров одной и той же строкой при удалении дубликатов, чем выше это значение, тем строже. Например, 0.99 означает, что из 100 символов 99 должны быть идентичны, чтобы считаться схожими
# Используется динамический алгоритм для определения порога схожести текста: для короткого текста требуется более низкий порог, для длинного текста - более высокий
# Например: для короткого текста "народ", "народ", 0.5 считается схожим
THRESHOLD_TEXT_SIMILARITY = 0.8

# Уверенность в извлечении субтитров ниже 0.75 отбрасывается
DROP_SCORE = 0.5

# Допустимое отклонение области субтитров, 0 - не допускается выход за границы, 0.03 означает, что можно выйти за границы на 3%
SUB_AREA_DEVIATION_RATE = 0.05

# Вывод потерянных кадров субтитров, действует только для упрощенного китайского, традиционного китайского, японского, корейского, по умолчанию отладочная информация выводится в: путь_к_видео/loss
DEBUG_OCR_LOSS = False

# Не удалять кэшированные данные для удобства отладки
DEBUG_NO_DELETE_CACHE = False

# Удалять ли пустые временные метки
DELETE_EMPTY_TIMESTAMP = True

# Выполнять ли повторную сегментацию слов, для решения проблемы отсутствия пробелов в предложениях
WORD_SEGMENTATION = True

# --------------------- Измените согласно вашей ситуации end-----------------------------

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
