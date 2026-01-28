# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao
@Time    : 2021/3/24 9:28 上午
@FileName: main.py
@desc: Главный файл входа в программу
"""
import os
import random
import shutil
from collections import Counter, namedtuple
import unicodedata
from threading import Thread
from pathlib import Path
import cv2
from Levenshtein import ratio
from PIL import Image
from numpy import average, dot, linalg
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(__file__))
import importlib
import config
from tools import reformat
from tools.infer import utility
from tools.infer.predict_det import TextDetector
from tools.ocr import OcrRecogniser, get_coordinates
from tools import subtitle_ocr
import threading
import platform
import multiprocessing
import time
import pysrt


class SubtitleDetect:
    """
    Класс детектирования текстовых блоков для обнаружения наличия субтитров в кадрах видео
    """

    def __init__(self):
        # Получение объекта параметров
        importlib.reload(config)
        args = utility.parse_args()
        args.det_algorithm = 'DB'
        args.det_model_dir = config.DET_MODEL_PATH
        self.text_detector = TextDetector(args)

    def detect_subtitle(self, img):
        dt_boxes, elapse = self.text_detector(img)
        return dt_boxes, elapse


class SubtitleExtractor:
    """
    Класс извлечения субтитров из видео
    """

    def __init__(self, vd_path, sub_area=None, gui_mode=False):
        importlib.reload(config)
        # Блокировка потока
        self.lock = threading.RLock()
        # Позиция области субтитров, указанная пользователем
        self.sub_area = sub_area
        # Создание объекта детектирования субтитров
        self.sub_detector = SubtitleDetect()
        # Путь к видео
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(vd_path)
        # Получение названия видео из пути
        self.vd_name = Path(self.video_path).stem
        # Временная папка для хранения
        if gui_mode:
            # В режиме GUI папка output находится в корне проекта
            self.temp_output_dir = os.path.join(os.path.dirname(os.path.dirname(config.BASE_DIR)), 'output', str(self.vd_name))
        else:
            # В режиме командной строки папка output находится в директории backend
            self.temp_output_dir = os.path.join(os.path.dirname(config.BASE_DIR), 'output', str(self.vd_name))
        # Общее количество кадров видео
        self.frame_count = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # Частота кадров видео (FPS)
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # Размеры видео
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # Область появления субтитров по умолчанию, если пользователь не указал
        self.default_subtitle_area = config.DEFAULT_SUBTITLE_AREA
        # Директория для хранения извлеченных кадров видео
        self.frame_output_dir = os.path.join(self.temp_output_dir, 'frames')
        # Директория для хранения извлеченных файлов субтитров
        self.subtitle_output_dir = os.path.join(self.temp_output_dir, 'subtitle')
        # Создание папок, если они не существуют
        if not os.path.exists(self.frame_output_dir):
            os.makedirs(self.frame_output_dir)
        if not os.path.exists(self.subtitle_output_dir):
            os.makedirs(self.subtitle_output_dir)
        # Определение использования VSF для извлечения субтитров
        self.use_vsf = False
        # Путь вывода субтитров VSF
        self.vsf_subtitle = os.path.join(self.subtitle_output_dir, 'raw_vsf.srt')
        # Путь хранения исходного текста субтитров
        self.raw_subtitle_path = os.path.join(self.subtitle_output_dir, 'raw.txt')
        # Пользовательский объект OCR
        self.ocr = None
        # Вывод языка распознавания и режима распознавания
        print(f"{config.interface_config['Main']['RecSubLang']}：{config.REC_CHAR_TYPE}")
        print(f"{config.interface_config['Main']['RecMode']}：{config.MODE_TYPE}")
        # Вывод подсказки об ускорении GPU, если используется
        if config.USE_GPU:
            print(config.interface_config['Main']['GPUSpeedUp'])
        # Общий прогресс обработки
        self.progress_total = 0
        # Прогресс извлечения кадров видео
        self.progress_frame_extract = 0
        # Прогресс OCR распознавания
        self.progress_ocr = 0
        # Флаг завершения
        self.isFinished = False
        # Очередь задач OCR субтитров
        self.subtitle_ocr_task_queue = None
        # Очередь прогресса OCR субтитров
        self.subtitle_ocr_progress_queue = None
        # Статус выполнения VSF
        self.vsf_running = False
        # Флаг режима GUI
        self.gui_mode = gui_mode

    def run(self):
        """
        Запуск всего процесса извлечения субтитров
        """
        # Записываем начальное время
        start_time = time.time()
        self.lock.acquire()
        
        # Сброс прогресса
        self.update_progress(ocr=0, frame_extract=0)
        
        # Вывод информации о видео
        print(f"{config.interface_config['Main']['FrameCount']}: {self.frame_count}"
              f", {config.interface_config['Main']['FrameRate']}: {self.fps}")
        
        # Вывод информации о моделях
        print(f'{os.path.basename(os.path.dirname(config.DET_MODEL_PATH))}-{os.path.basename(config.DET_MODEL_PATH)}')
        print(f'{os.path.basename(os.path.dirname(config.REC_MODEL_PATH))}-{os.path.basename(config.REC_MODEL_PATH)}')
        
        # Проверяем использование GPU и при необходимости переключаемся на CPU
        if config.USE_GPU:
            print(config.interface_config['Main']['GPUSpeedUp'])
        else:
            print("Используется CPU-режим")
            
        # Принудительный переход на fast режим, если VSF не работает
        if self.sub_area is not None and platform.system() == 'Linux':
            print("Linux система, проверяем совместимость VSF...")
            # Если у нас проблемы с VSF, используем fast режим вместо VSF
            if hasattr(self, 'vsf_failed') and self.vsf_failed:
                print("VSF недоступен, используем метод извлечения по кадрам")
                self.sub_area = None  # Отключаем указание области для использования метода по кадрам
        
        print(config.interface_config['Main']['StartProcessFrame'])
        
        # Создаем процесс OCR распознавания субтитров
        subtitle_ocr_process = self.start_subtitle_ocr_async()
        
        # Выбор метода извлечения кадров
        if self.sub_area is not None:
            if platform.system() in ['Windows', 'Linux']:
                # Пробуем использовать VSF
                try:
                    self.extract_frame_by_vsf()
                    # Проверяем, создал ли VSF файлы
                    if self.use_vsf and os.path.exists(self.vsf_subtitle):
                        with open(self.vsf_subtitle, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if len(content.strip()) == 0:
                                print("VSF создал пустой файл, переключаюсь на метод по кадрам")
                                self.use_vsf = False
                                self.extract_frame_by_fps()
                    elif self.use_vsf:
                        print("VSF не создал файл субтитров, переключаюсь на метод по кадрам")
                        self.use_vsf = False
                        self.extract_frame_by_fps()
                except Exception as e:
                    print(f"Ошибка при использовании VSF: {e}")
                    print("Переключаюсь на метод извлечения по кадрам")
                    self.use_vsf = False
                    self.extract_frame_by_fps()
            else:
                # Для других систем используем метод по кадрам
                self.extract_frame_by_fps()
        else:
            # Если область субтитров не указана, используем метод по кадрам
            self.extract_frame_by_fps()
        
        # Отправляем сигнал завершения в очередь задач OCR
        self.subtitle_ocr_task_queue.put((self.frame_count, -1, None, None, None, None))
        
        # Ожидаем завершения процесса OCR
        subtitle_ocr_process.join()
        
        print(config.interface_config['Main']['FinishProcessFrame'])
        print(config.interface_config['Main']['FinishFindSub'])
        
        # Проверяем, создался ли raw файл с субтитрами
        if not os.path.exists(self.raw_subtitle_path) or os.path.getsize(self.raw_subtitle_path) == 0:
            print(f"ОШИБКА: Файл {self.raw_subtitle_path} пустой или не существует!")
            print("Проверьте, что видео содержит субтитры и область указана правильно.")
            print("Попробуйте указать другую область субтитров.")
            self.lock.release()
            return
        
        # Вопрос о водяных знаках (только если область не указана)
        if self.sub_area is None:
            print(config.interface_config['Main']['StartDetectWaterMark'])
            user_input = input(config.interface_config['Main']['checkWaterMark']).strip()
            if user_input == 'y':
                self.filter_watermark()
                print(config.interface_config['Main']['FinishDetectWaterMark'])
            else:
                print('-----------------------------')
        
        # Фильтрация текста сцены (только если область не указана)
        if self.sub_area is None:
            print(config.interface_config['Main']['StartDeleteNonSub'])
            self.filter_scene_text()
            print(config.interface_config['Main']['FinishDeleteNonSub'])
        
        # Генерация файла субтитров
        print(config.interface_config['Main']['StartGenerateSub'])
        
        if self.use_vsf and os.path.exists(self.vsf_subtitle) and os.path.getsize(self.vsf_subtitle) > 0:
            self.generate_subtitle_file_vsf()
        else:
            self.generate_subtitle_file()
        
        if config.WORD_SEGMENTATION:
            reformat.execute(os.path.join(os.path.splitext(self.video_path)[0] + '.srt'), config.REC_CHAR_TYPE)
        
        print(f"{config.interface_config['Main']['FinishGenerateSub']} за {round(time.time() - start_time, 2)} секунд")
        
        # Проверяем, создался ли итоговый файл субтитров
        srt_file = os.path.join(os.path.splitext(self.video_path)[0] + '.srt')
        if os.path.exists(srt_file) and os.path.getsize(srt_file) > 0:
            print(f"Субтитры успешно созданы: {srt_file}")
        else:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Файл субтитров {srt_file} пустой или не создан!")
            print("Возможные причины:")
            print("1. В видео нет субтитров в указанной области")
            print("2. Неправильно указана область субтитров")
            print("3. Слишком высокий порог уверенности DROP_SCORE")
            print("4. Проблемы с моделью распознавания")
        
        self.update_progress(ocr=100, frame_extract=100)
        self.isFinished = True
        
        # Очистка кэша
        self.empty_cache()
        self.lock.release()
        
        # Создание TXT файла, если нужно
        if config.GENERATE_TXT:
            self.srt2txt(srt_file)

    def extract_frame_by_vsf(self):
        """
        Извлечение субтитровых кадров через вызов VideoSubFinder
        """
        self.use_vsf = True

        def vsf_output(out):
            duration_ms = (self.frame_count / self.fps) * 1000
            last_total_ms = 0
            for line in iter(out.readline, b''):
                line = line.decode("utf-8", errors='ignore')
                if line.startswith('Frame: '):
                    line = line.replace("\n", "")
                    line = line.replace("Frame: ", "")
                    try:
                        h, m, s, ms = line.split('__')[0].split('_')
                        total_ms = int(ms) + int(s) * 1000 + int(m) * 60 * 1000 + int(h) * 60 * 60 * 1000
                        if total_ms > last_total_ms:
                            frame_no = int(total_ms / self.fps)
                            task = (self.frame_count, frame_no, None, None, total_ms, self.default_subtitle_area)
                            self.subtitle_ocr_task_queue.put(task)
                        last_total_ms = total_ms
                        if total_ms / duration_ms >= 1:
                            self.update_progress(frame_extract=100)
                            return
                        else:
                            self.update_progress(frame_extract=(total_ms / duration_ms) * 100)
                    except:
                        continue
                else:
                    print(line.strip())
            out.close()

        # Удаляем кэш кадров
        self.__delete_frame_cache()
        
        # Определяем путь к VideoSubFinder
        if platform.system() == 'Windows':
            path_vsf = os.path.join(config.BASE_DIR, 'subfinder', 'windows', 'VideoSubFinderWXW.exe')
        else:
            path_vsf = os.path.join(config.BASE_DIR, 'subfinder', 'linux', 'VideoSubFinderCli')
            if not os.path.exists(path_vsf):
                path_vsf = os.path.join(config.BASE_DIR, 'subfinder', 'linux', 'VideoSubFinderCli.run')
        
        # Проверяем существование файла
        if not os.path.exists(path_vsf):
            print(f"VideoSubFinder не найден по пути: {path_vsf}")
            print("Переключаюсь на метод извлечения по кадрам")
            self.use_vsf = False
            self.extract_frame_by_fps()
            return

        # Проверяем, можно ли запустить VSF
        try:
            # Пробный запуск без параметров
            test_cmd = f'"{path_vsf}" --help' if platform.system() == 'Windows' else f'"{path_vsf}" --help'
            import subprocess
            result = subprocess.run(test_cmd, shell=True, capture_output=True, timeout=5)
            
            # Если VSF возвращает ошибку или segfault, переключаемся на CPU
            if result.returncode != 0:
                print(f"VSF возвращает код ошибки {result.returncode}")
                print("Переключаюсь на метод извлечения по кадрам")
                self.use_vsf = False
                self.extract_frame_by_fps()
                return
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as e:
            print(f"VSF не может быть запущен: {e}")
            print("Переключаюсь на метод извлечения по кадрам")
            self.use_vsf = False
            self.extract_frame_by_fps()
            return
        
        # Параметры для области субтитров
        top_end = 1 - self.sub_area[0] / self.frame_height
        bottom_end = 1 - self.sub_area[1] / self.frame_height
        left_end = self.sub_area[2] / self.frame_width
        right_end = self.sub_area[3] / self.frame_width
        
        cpu_count = max(int(multiprocessing.cpu_count() * 2 / 3), 1)
        if cpu_count < 4:
            cpu_count = max(multiprocessing.cpu_count() - 1, 1)
        
        try:
            if platform.system() == 'Windows':
                cmd = f'"{path_vsf}" --use_cuda -c -r -i "{self.video_path}" -o "{self.temp_output_dir}" -ces "{self.vsf_subtitle}" '
                cmd += f'-te {top_end} -be {bottom_end} -le {left_end} -re {right_end} -nthr {cpu_count} -nocrthr {cpu_count}'
            else:
                # Для Linux используем прямой вызов бинарного файла
                cmd = f'"{path_vsf}" -c -r -i "{self.video_path}" -o "{self.temp_output_dir}" -ces "{self.vsf_subtitle}" '
                cmd += f'-te {top_end} -be {bottom_end} -le {left_end} -re {right_end} -nthr {cpu_count}'
            
            print(f"Запуск VSF командой: {cmd[:200]}...")
            
            self.vsf_running = True
            import subprocess
            
            # Запускаем с таймаутом
            process = subprocess.Popen(
                cmd, 
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Ждем завершения с таймаутом
            timeout_seconds = 30  # Таймаут 30 секунд
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
                
                if process.returncode != 0:
                    print(f"VSF завершился с ошибкой кодом {process.returncode}")
                    print(f"stderr: {stderr[:500]}")
                    print("Переключаюсь на метод извлечения по кадрам")
                    self.use_vsf = False
                    self.extract_frame_by_fps()
                    return
                    
                print("VSF успешно завершил работу")
                
            except subprocess.TimeoutExpired:
                print(f"VSF превысил таймаут {timeout_seconds} секунд")
                process.kill()
                print("Переключаюсь на метод извлечения по кадрам")
                self.use_vsf = False
                self.extract_frame_by_fps()
                return
                
        except Exception as e:
            print(f"Ошибка при запуске VSF: {e}")
            import traceback
            traceback.print_exc()
            print("Переключаюсь на метод извлечения по кадрам")
            self.use_vsf = False
            self.extract_frame_by_fps()
            return
        
        self.vsf_running = False

    def extract_frame_by_fps(self):
        """
        Извлечение кадров по частоте X кадров в секунду
        """
        # Удаление кэша
        self.__delete_frame_cache()

        # Номер текущего кадра видео
        current_frame_no = 0
        frame_lru_list = []
        frame_lru_list_max_size = 2
        ocr_args_list = []
        compare_ocr_result_cache = {}
        tbar = tqdm(total=int(self.frame_count), unit='f', position=0, file=sys.__stdout__)
        first_flag = True
        is_finding_start_frame_no = False
        is_finding_end_frame_no = False
        start_frame_no = 0
        start_end_frame_no = []
        start_frame = None
        if self.ocr is None:
            self.ocr = OcrRecogniser()
        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            # Если чтение кадра не удалось (конец видео)
            if not ret:
                break
            # Успешное чтение кадра
            current_frame_no += 1
            tbar.update(1)
            # X кадров в секунду
            if current_frame_no % int(self.fps / config.EXTRACT_FREQUENCY) == 0:
                # По умолчанию предполагаем наличие субтитров
                has_subtitle = True
                # Обнаружение начального и конечного номера кадра, содержащего субтитры
                if has_subtitle:
                    # Определяем, является ли кадр начальным или конечным
                    if is_finding_start_frame_no:
                        start_frame_no = current_frame_no
                        dt_box, rec_res = self.ocr.predict(frame)
                        area_text1 = "".join(self.__get_area_text((dt_box, rec_res)))
                        if start_frame_no not in compare_ocr_result_cache.keys():
                            compare_ocr_result_cache[current_frame_no] = {'text': area_text1, 'dt_box': dt_box, 'rec_res': rec_res}
                            frame_lru_list.append((frame, current_frame_no))
                            ocr_args_list.append((self.frame_count, current_frame_no))
                            # Кэшируем начальный кадр
                            start_frame = frame
                        # Начинаем поиск конечного кадра
                        is_finding_start_frame_no = False
                        is_finding_end_frame_no = True
                    # Определяем, является ли кадр последним
                    if is_finding_end_frame_no and current_frame_no == self.frame_count:
                        is_finding_end_frame_no = False
                        is_finding_start_frame_no = False
                        end_frame_no = current_frame_no
                        frame_lru_list.append((frame, current_frame_no))
                        ocr_args_list.append((self.frame_count, current_frame_no))
                        start_end_frame_no.append((start_frame_no, end_frame_no))
                    # Если находимся в поиске конечного кадра
                    if is_finding_end_frame_no:
                        # Проверяем, совпадает ли содержимое OCR этого кадра с начальным кадром. Если нет, то найден конечный кадр (предыдущий кадр)
                        if not self._compare_ocr_result(compare_ocr_result_cache, None, start_frame_no, frame, current_frame_no):
                            is_finding_end_frame_no = False
                            is_finding_start_frame_no = True
                            end_frame_no = current_frame_no - 1
                            frame_lru_list.append((start_frame, end_frame_no))
                            ocr_args_list.append((self.frame_count, end_frame_no))
                            start_end_frame_no.append((start_frame_no, end_frame_no))

                else:
                    # Если после обнаружения начального кадра субтитров нет, то найден конечный кадр (предыдущий кадр)
                    if is_finding_end_frame_no:
                        end_frame_no = current_frame_no - 1
                        is_finding_end_frame_no = False
                        is_finding_start_frame_no = True
                        frame_lru_list.append((start_frame, end_frame_no))
                        ocr_args_list.append((self.frame_count, end_frame_no))
                        start_end_frame_no.append((start_frame_no, end_frame_no))

                while len(frame_lru_list) > frame_lru_list_max_size:
                    frame_lru_list.pop(0)

                # if len(start_end_frame_no) > 0:
                    # print(start_end_frame_no)

                while len(ocr_args_list) > 1:
                    total_frame_count, ocr_info_frame_no = ocr_args_list.pop(0)
                    if current_frame_no in compare_ocr_result_cache:
                        predict_result = compare_ocr_result_cache[current_frame_no]
                        dt_box, rec_res = predict_result['dt_box'], predict_result['rec_res']
                    else:
                        dt_box, rec_res = None, None
                    # subtitle_ocr_task_queue: (total_frame_count общее количество кадров, current_frame_no текущий кадр, dt_box ограничивающая рамка, rec_res результат распознавания, время текущего кадра, subtitle_area область субтитров)
                    task = (total_frame_count, ocr_info_frame_no, dt_box, rec_res, None, self.default_subtitle_area)
                    # Добавление задачи
                    self.subtitle_ocr_task_queue.put(task)
                    self.update_progress(frame_extract=(current_frame_no / self.frame_count) * 100)

        while len(ocr_args_list) > 0:
            total_frame_count, ocr_info_frame_no = ocr_args_list.pop(0)
            if current_frame_no in compare_ocr_result_cache:
                predict_result = compare_ocr_result_cache[current_frame_no]
                dt_box, rec_res = predict_result['dt_box'], predict_result['rec_res']
            else:
                dt_box, rec_res = None, None
            task = (total_frame_count, ocr_info_frame_no, dt_box, rec_res, None, self.default_subtitle_area)
            # Добавление задачи
            self.subtitle_ocr_task_queue.put(task)
        self.video_cap.release()

    def extract_frame_by_det(self):
        """
        Извлечение кадров субтитров через обнаружение позиции области субтитров
        """
        # Удаление кэша
        self.__delete_frame_cache()

        # Номер текущего кадра видео
        current_frame_no = 0
        frame_lru_list = []
        frame_lru_list_max_size = 2
        ocr_args_list = []
        compare_ocr_result_cache = {}
        tbar = tqdm(total=int(self.frame_count), unit='f', position=0, file=sys.__stdout__)
        first_flag = True
        is_finding_start_frame_no = False
        is_finding_end_frame_no = False
        start_frame_no = 0
        start_end_frame_no = []
        start_frame = None
        if self.ocr is None:
            self.ocr = OcrRecogniser()
        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            # Если чтение кадра не удалось (конец видео)
            if not ret:
                break
            # Успешное чтение кадра
            current_frame_no += 1
            tbar.update(1)
            dt_boxes, elapse = self.sub_detector.detect_subtitle(frame)
            has_subtitle = False
            if self.sub_area is not None:
                s_ymin, s_ymax, s_xmin, s_xmax = self.sub_area
                coordinate_list = get_coordinates(dt_boxes.tolist())
                if coordinate_list:
                    for coordinate in coordinate_list:
                        xmin, xmax, ymin, ymax = coordinate
                        if (s_xmin <= xmin and xmax <= s_xmax
                                and s_ymin <= ymin
                                and ymax <= s_ymax):
                            has_subtitle = True
                            # При обнаружении субтитров, если список пуст, это начальный кадр
                            if first_flag:
                                is_finding_start_frame_no = True
                                first_flag = False
                            break
            else:
                has_subtitle = len(dt_boxes) > 0
            # Обнаружение начального и конечного номера кадра, содержащего субтитры
            if has_subtitle:
                # Определяем, является ли кадр начальным или конечным
                if is_finding_start_frame_no:
                    start_frame_no = current_frame_no
                    dt_box, rec_res = self.ocr.predict(frame)
                    area_text1 = "".join(self.__get_area_text((dt_box, rec_res)))
                    if start_frame_no not in compare_ocr_result_cache.keys():
                        compare_ocr_result_cache[current_frame_no] = {'text': area_text1, 'dt_box': dt_box, 'rec_res': rec_res}
                        frame_lru_list.append((frame, current_frame_no))
                        ocr_args_list.append((self.frame_count, current_frame_no))
                        # Кэшируем начальный кадр
                        start_frame = frame
                    # Начинаем поиск конечного кадра
                    is_finding_start_frame_no = False
                    is_finding_end_frame_no = True
                # Определяем, является ли кадр последним
                if is_finding_end_frame_no and current_frame_no == self.frame_count:
                    is_finding_end_frame_no = False
                    is_finding_start_frame_no = False
                    end_frame_no = current_frame_no
                    frame_lru_list.append((frame, current_frame_no))
                    ocr_args_list.append((self.frame_count, current_frame_no))
                    start_end_frame_no.append((start_frame_no, end_frame_no))
                # Если находимся в поиске конечного кадра
                if is_finding_end_frame_no:
                    # Проверяем, совпадает ли содержимое OCR этого кадра с начальным кадром. Если нет, то найден конечный кадр (предыдущий кадр)
                    if not self._compare_ocr_result(compare_ocr_result_cache, None, start_frame_no, frame, current_frame_no):
                        is_finding_end_frame_no = False
                        is_finding_start_frame_no = True
                        end_frame_no = current_frame_no - 1
                        frame_lru_list.append((start_frame, end_frame_no))
                        ocr_args_list.append((self.frame_count, end_frame_no))
                        start_end_frame_no.append((start_frame_no, end_frame_no))

            else:
                # Если после обнаружения начального кадра субтитров нет, то найден конечный кадр (предыдущий кадр)
                if is_finding_end_frame_no:
                    end_frame_no = current_frame_no - 1
                    is_finding_end_frame_no = False
                    is_finding_start_frame_no = True
                    frame_lru_list.append((start_frame, end_frame_no))
                    ocr_args_list.append((self.frame_count, end_frame_no))
                    start_end_frame_no.append((start_frame_no, end_frame_no))

            while len(frame_lru_list) > frame_lru_list_max_size:
                frame_lru_list.pop(0)

            # if len(start_end_frame_no) > 0:
                # print(start_end_frame_no)

            while len(ocr_args_list) > 1:
                total_frame_count, ocr_info_frame_no = ocr_args_list.pop(0)
                if current_frame_no in compare_ocr_result_cache:
                    predict_result = compare_ocr_result_cache[current_frame_no]
                    dt_box, rec_res = predict_result['dt_box'], predict_result['rec_res']
                else:
                    dt_box, rec_res = None, None
                # subtitle_ocr_task_queue: (total_frame_count общее количество кадров, current_frame_no текущий кадр, dt_box ограничивающая рамка, rec_res результат распознавания, время текущего кадра, subtitle_area область субтитров)
                task = (total_frame_count, ocr_info_frame_no, dt_box, rec_res, None, self.default_subtitle_area)
                # Добавление задачи
                self.subtitle_ocr_task_queue.put(task)
                self.update_progress(frame_extract=(current_frame_no / self.frame_count) * 100)

        while len(ocr_args_list) > 0:
            total_frame_count, ocr_info_frame_no = ocr_args_list.pop(0)
            if current_frame_no in compare_ocr_result_cache:
                predict_result = compare_ocr_result_cache[current_frame_no]
                dt_box, rec_res = predict_result['dt_box'], predict_result['rec_res']
            else:
                dt_box, rec_res = None, None
            task = (total_frame_count, ocr_info_frame_no, dt_box, rec_res, None, self.default_subtitle_area)
            # Добавление задачи
            self.subtitle_ocr_task_queue.put(task)
        self.video_cap.release()

    def filter_watermark(self):
        """
        Удаление текста из области водяного знака в исходном тексте субтитров
        """
        # Получение потенциальных областей водяных знаков
        watermark_areas = self._detect_watermark_area()

        # Случайный выбор кадра для маркировки областей водяных знаков, пользователь определяет, является ли это областью водяного знака
        cap = cv2.VideoCapture(self.video_path)
        ret, sample_frame = False, None
        for i in range(10):
            frame_no = random.randint(int(self.frame_count * 0.1), int(self.frame_count * 0.9))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, sample_frame = cap.read()
            if ret:
                break
        cap.release()

        if not ret:
            print("Ошибка в filter_watermark: чтение кадра из видео")
            return

        # Нумерация потенциальных областей водяных знаков
        area_num = ['E', 'D', 'C', 'B', 'A']

        for watermark_area in watermark_areas:
            ymin = min(watermark_area[0][2], watermark_area[0][3])
            ymax = max(watermark_area[0][3], watermark_area[0][2])
            xmin = min(watermark_area[0][0], watermark_area[0][1])
            xmax = max(watermark_area[0][1], watermark_area[0][0])
            cover = sample_frame[ymin:ymax, xmin:xmax]
            cover = cv2.blur(cover, (10, 10))
            cv2.rectangle(cover, pt1=(0, cover.shape[0]), pt2=(cover.shape[1], 0), color=(0, 0, 255), thickness=3)
            sample_frame[ymin:ymax, xmin:xmax] = cover
            position = ((xmin + xmax) // 2, ymax)

            cv2.putText(sample_frame, text=area_num.pop(), org=position, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        sample_frame_file_path = os.path.join(os.path.dirname(self.frame_output_dir), 'watermark_area.jpg')
        cv2.imwrite(sample_frame_file_path, sample_frame)
        print(f"{config.interface_config['Main']['WatchPicture']}: {sample_frame_file_path}")

        area_num = ['E', 'D', 'C', 'B', 'A']
        for watermark_area in watermark_areas:
            user_input = input(f"{area_num.pop()}{str(watermark_area)} "
                               f"{config.interface_config['Main']['QuestionDelete']}").strip()
            if user_input == 'y' or user_input == '\n':
                with open(self.raw_subtitle_path, mode='r+', encoding='utf-8') as f:
                    content = f.readlines()
                    f.seek(0)
                    for i in content:
                        if i.find(str(watermark_area[0])) == -1:
                            f.write(i)
                    f.truncate()
                print(config.interface_config['Main']['FinishDelete'])
        print(config.interface_config['Main']['FinishWaterMarkFilter'])
        # Удаление кэша
        if os.path.exists(sample_frame_file_path):
            os.remove(sample_frame_file_path)

    def filter_scene_text(self):
        """
        Фильтрация текста, извлеченного из сцены, сохранение только области субтитров
        """
        # Получение потенциальной области субтитров
        subtitle_area = self._detect_subtitle_area()[0][0]

        # Случайный выбор кадра для маркировки области, пользователь определяет, является ли это областью водяного знака
        cap = cv2.VideoCapture(self.video_path)
        ret, sample_frame = False, None
        for i in range(10):
            frame_no = random.randint(int(self.frame_count * 0.1), int(self.frame_count * 0.9))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, sample_frame = cap.read()
            if ret:
                break
        cap.release()

        if not ret:
            print("Ошибка в filter_scene_text: чтение кадра из видео")
            return

        # Для учета двойных строк субтитров увеличиваем диапазон области субтитров по оси Y в соответствии с допуском
        ymin = abs(subtitle_area[0] - config.SUBTITLE_AREA_DEVIATION_PIXEL)
        ymax = subtitle_area[1] + config.SUBTITLE_AREA_DEVIATION_PIXEL
        # Отрисовка области субтитров
        cv2.rectangle(sample_frame, pt1=(0, ymin), pt2=(sample_frame.shape[1], ymax), color=(0, 0, 255), thickness=3)
        sample_frame_file_path = os.path.join(os.path.dirname(self.frame_output_dir), 'subtitle_area.jpg')
        cv2.imwrite(sample_frame_file_path, sample_frame)
        print(f"{config.interface_config['Main']['CheckSubArea']} {sample_frame_file_path}")

        user_input = input(f"{(ymin, ymax)} {config.interface_config['Main']['DeleteNoSubArea']}").strip()
        if user_input == 'y' or user_input == '\n':
            with open(self.raw_subtitle_path, mode='r+', encoding='utf-8') as f:
                content = f.readlines()
                f.seek(0)
                for i in content:
                    i_ymin = int(i.split('\t')[1].split('(')[1].split(')')[0].split(', ')[2])
                    i_ymax = int(i.split('\t')[1].split('(')[1].split(')')[0].split(', ')[3])
                    if ymin <= i_ymin and i_ymax <= ymax:
                        f.write(i)
                f.truncate()
            print(config.interface_config['Main']['FinishDeleteNoSubArea'])
        # Удаление кэша
        if os.path.exists(sample_frame_file_path):
            os.remove(sample_frame_file_path)

    def generate_subtitle_file(self):
        """
        Генерация файла субтитров в формате SRT
        """
        if not self.use_vsf:
            subtitle_content = self._remove_duplicate_subtitle()
            srt_filename = os.path.join(os.path.splitext(self.video_path)[0] + '.srt')
            # Сохранение строк субтитров с длительностью менее 1 секунды для последующей обработки
            post_process_subtitle = []
            with open(srt_filename, mode='w', encoding='utf-8') as f:
                for index, content in enumerate(subtitle_content):
                    line_code = index + 1
                    frame_start = self._frame_to_timecode(int(content[0]))
                    # Сравнение начального и конечного номера кадра, если длительность субтитров менее 1 секунды, устанавливаем время отображения 1 с
                    if abs(int(content[1]) - int(content[0])) < self.fps:
                        frame_end = self._frame_to_timecode(int(int(content[0]) + self.fps))
                        post_process_subtitle.append(line_code)
                    else:
                        frame_end = self._frame_to_timecode(int(content[1]))
                    frame_content = content[2]
                    subtitle_line = f'{line_code}\n{frame_start} --> {frame_end}\n{frame_content}\n'
                    f.write(subtitle_line)
            print(f"[NO-VSF]{config.interface_config['Main']['SubLocation']} {srt_filename}")
            # Возврат строк субтитров с длительностью менее 1 с
            return post_process_subtitle

    def generate_subtitle_file_vsf(self):
        if not self.use_vsf:
            return
        subs = pysrt.open(self.vsf_subtitle)
        sub_no_map = {}
        for sub in subs:
            sub.start.no = self._timestamp_to_frameno(sub.start.ordinal)
            sub_no_map[sub.start.no] = sub

        subtitle_content = self._remove_duplicate_subtitle()
        subtitle_content_start_map = {int(a[0]): a for a in subtitle_content}
        final_subtitles = []
        for sub in subs:
            found = sub.start.no in subtitle_content_start_map
            if found:
                subtitle_content_line = subtitle_content_start_map[sub.start.no]
                sub.text = subtitle_content_line[2]
                end_no = int(subtitle_content_line[1])
                sub.end = sub_no_map[end_no].end if end_no in sub_no_map else sub.end
                sub.index = len(final_subtitles) + 1
                final_subtitles.append(sub)

            if not found and not config.DELETE_EMPTY_TIMESTAMP:
                # Сохраняем временную шкалу
                sub.text = ""
                sub.index = len(final_subtitles) + 1
                final_subtitles.append(sub)
                continue

        srt_filename = os.path.join(os.path.splitext(self.video_path)[0] + '.srt')
        pysrt.SubRipFile(final_subtitles).save(srt_filename, encoding='utf-8')
        print(f"[VSF]{config.interface_config['Main']['SubLocation']} {srt_filename}")

    def _detect_watermark_area(self):
        """
        Поиск области водяного знака на основе информации о координатах в сыром txt файле
        Предположение: координаты области водяного знака (логотипа) фиксированы по горизонтали и вертикали, т.е. (xmin, xmax, ymin, ymax) относительно постоянны
        На основе информации о координатах выполняется статистика для выбора текстовых областей с фиксированными координатами
        :return Возвращает наиболее вероятную область водяного знака
        """
        f = open(self.raw_subtitle_path, mode='r', encoding='utf-8')  # Открытие txt файла с кодировкой 'utf-8'
        line = f.readline()  # Чтение файла построчно
        # Список координат
        coordinates_list = []
        # Список номеров кадров
        frame_no_list = []
        # Список содержимого
        content_list = []
        while line:
            frame_no = line.split('\t')[0]
            text_position = line.split('\t')[1].split('(')[1].split(')')[0].split(', ')
            content = line.split('\t')[2]
            frame_no_list.append(frame_no)
            coordinates_list.append((int(text_position[0]),
                                     int(text_position[1]),
                                     int(text_position[2]),
                                     int(text_position[3])))
            content_list.append(content)
            line = f.readline()
        f.close()
        # Унификация похожих значений в списке координат
        coordinates_list = self._unite_coordinates(coordinates_list)

        # Обновление координат в исходном txt файле на нормализованные
        with open(self.raw_subtitle_path, mode='w', encoding='utf-8') as f:
            for frame_no, coordinate, content in zip(frame_no_list, coordinates_list, content_list):
                f.write(f'{frame_no}\t{coordinate}\t{content}')

        if len(Counter(coordinates_list).most_common()) > config.WATERMARK_AREA_NUM:
            # Чтение конфигурации, возврат списка координат, которые могут быть областью водяного знака
            return Counter(coordinates_list).most_common(config.WATERMARK_AREA_NUM)
        else:
            # Если недостаточно, возвращаем столько, сколько есть
            return Counter(coordinates_list).most_common()

    def _detect_subtitle_area(self):
        """
        Чтение сырого txt файла после фильтрации области водяного знака, поиск области субтитров на основе информации о координатах
        Предположение: область субтитров имеет относительно фиксированный диапазон координат по оси Y, по сравнению с текстом сцены, этот диапазон встречается чаще
        :return Возвращает позицию области субтитров
        """
        # Открытие обработанного сырого txt файла с удаленной областью водяного знака
        f = open(self.raw_subtitle_path, mode='r', encoding='utf-8')  # Открытие txt файла с кодировкой 'utf-8'
        line = f.readline()  # Чтение файла построчно
        # Список Y-координат
        y_coordinates_list = []
        while line:
            text_position = line.split('\t')[1].split('(')[1].split(')')[0].split(', ')
            y_coordinates_list.append((int(text_position[2]), int(text_position[3])))
            line = f.readline()
        f.close()
        return Counter(y_coordinates_list).most_common(1)

    def _frame_to_timecode(self, frame_no):
        """
        Преобразование номера кадра видео во временную метку
        :param frame_no: Номер кадра видео, т.е. какой по счету кадр
        :returns: Временная метка в формате SMPTE в виде строки, например '01:02:12:032' или '01:02:12;032'
        """
        # Установка текущего номера кадра
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, _ = cap.read()
        # Получение временной метки, соответствующей текущему номеру кадра
        if ret:
            milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
            if milliseconds <= 0:
                return '{0:02d}:{1:02d}:{2:02d},{3:03d}'.format(int(frame_no / (3600 * self.fps)),
                                                                int(frame_no / (60 * self.fps) % 60),
                                                                int(frame_no / self.fps % 60),
                                                                int(frame_no % self.fps))
            seconds = milliseconds // 1000
            milliseconds = int(milliseconds % 1000)
            minutes = 0
            hours = 0
            if seconds >= 60:
                minutes = int(seconds // 60)
                seconds = int(seconds % 60)
            if minutes >= 60:
                hours = int(minutes // 60)
                minutes = int(minutes % 60)
            smpte_token = ','
            cap.release()
            return "%02d:%02d:%02d%s%03d" % (hours, minutes, seconds, smpte_token, milliseconds)
        else:
            return '{0:02d}:{1:02d}:{2:02d},{3:03d}'.format(int(frame_no / (3600 * self.fps)),
                                                            int(frame_no / (60 * self.fps) % 60),
                                                            int(frame_no / self.fps % 60),
                                                            int(frame_no % self.fps))

    def _timestamp_to_frameno(self, time_ms):
        return int(time_ms / self.fps)

    def _frameno_to_milliseconds(self, frame_no):
        return float(int(frame_no / self.fps * 1000))

    def _remove_duplicate_subtitle(self):
        """
        Чтение исходного сырого txt, удаление повторяющихся строк, возврат списка субтитров после дедупликации
        """
        self._concat_content_with_same_frameno()
        with open(self.raw_subtitle_path, mode='r', encoding='utf-8') as r:
            lines = r.readlines()
        RawInfo = namedtuple('RawInfo', 'no content')
        content_list = []
        for line in lines:
            frame_no = line.split('\t')[0]
            content = line.split('\t')[2]
            content_list.append(RawInfo(frame_no, content))
        # Список уникальных субтитров
        unique_subtitle_list = []
        idx_i = 0
        content_list_len = len(content_list)
        # Итерация по каждой строке субтитров, запись времени начала и окончания
        while idx_i < content_list_len:
            i = content_list[idx_i]
            start_frame = i.no
            idx_j = idx_i
            while idx_j < content_list_len:
                # Вычисление расстояния Левенштейна между текущей и следующей строкой
                # Проверка, отличается ли следующий кадр после idx_j от idx_i, если отличается (или это последний кадр), то найден конечный кадр
                if idx_j + 1 == content_list_len or ratio(i.content.replace(' ', ''), content_list[idx_j + 1].content.replace(' ', '')) < config.THRESHOLD_TEXT_SIMILARITY:
                    # Если найден конечный кадр, определяем номер конечного кадра субтитров
                    end_frame = content_list[idx_j].no
                    if not self.use_vsf:
                        if end_frame == start_frame and idx_j + 1 < content_list_len:
                            # Для случая только одного кадра используем время начала следующего кадра (если это не последний кадр)
                            end_frame = content_list[idx_j + 1][0]
                    # Поиск наиболее длинных субтитров
                    similar_list = content_list[idx_i:idx_j + 1]
                    similar_content_strip_list = [item.content.replace(' ', '') for item in similar_list]
                    index, _ = max(enumerate(similar_content_strip_list), key=lambda x: len(x[1]))

                    # Добавление в список
                    unique_subtitle_list.append((start_frame, end_frame, similar_list[index].content))
                    idx_i = idx_j + 1
                    break
                else:
                    idx_j += 1
                    continue
        return unique_subtitle_list

    def _concat_content_with_same_frameno(self):
        """
        Объединение строк субтитров с одинаковым номером кадра в сыром txt файле
        """
        with open(self.raw_subtitle_path, mode='r', encoding='utf-8') as r:
            lines = r.readlines()
        content_list = []
        frame_no_list = []
        for line in lines:
            frame_no = line.split('\t')[0]
            frame_no_list.append(frame_no)
            coordinate = line.split('\t')[1]
            content = line.split('\t')[2]
            content_list.append([frame_no, coordinate, content])

        # Нахождение номеров кадров, которые встречаются более одного раза
        frame_no_list = [i[0] for i in Counter(frame_no_list).most_common() if i[1] > 1]

        # Нахождение позиций, где встречаются эти номера кадров
        concatenation_list = []
        for frame_no in frame_no_list:
            position = [i for i, x in enumerate(content_list) if x[0] == frame_no]
            concatenation_list.append((frame_no, position))

        for i in concatenation_list:
            content = []
            for j in i[1]:
                content.append(content_list[j][2])
            content = ' '.join(content).replace('\n', ' ') + '\n'
            for k in i[1]:
                content_list[k][2] = content

        # Удаление лишних строк субтитров
        to_delete = []
        for i in concatenation_list:
            for j in i[1][1:]:
                to_delete.append(content_list[j])
        for i in to_delete:
            if i in content_list:
                content_list.remove(i)

        with open(self.raw_subtitle_path, mode='w', encoding='utf-8') as f:
            for frame_no, coordinate, content in content_list:
                content = unicodedata.normalize('NFKC', content)
                f.write(f'{frame_no}\t{coordinate}\t{content}')

    def _unite_coordinates(self, coordinates_list):
        """
        Унификация похожих координат в списке координат до одного значения
        Например, из-за того, что результаты обнаружения ограничивающих рамок непостоянны, координаты текста в одном и том же месте могут быть обнаружены как (255,123,456,789) в один раз и как (253,122,456,799) в другой
        Поэтому необходимо унифицировать значения похожих координат
        :param coordinates_list Список, содержащий точки координат
        :return: Возвращает список координат с унифицированными значениями
        """
        # Унификация похожих координат в одну
        index = 0
        for coordinate in coordinates_list:  # TODO: Сложность O(n^2), требуется оптимизация
            for i in coordinates_list:
                if self.__is_coordinate_similar(coordinate, i):
                    coordinates_list[index] = i
            index += 1
        return coordinates_list

    def _compute_image_similarity(self, image1, image2):
        """
        Вычисление косинусного сходства между двумя изображениями
        """
        image1 = self.__get_thum(image1)
        image2 = self.__get_thum(image2)
        images = [image1, image2]
        vectors = []
        norms = []
        for image in images:
            vector = []
            for pixel_tuple in image.getdata():
                vector.append(average(pixel_tuple))
            vectors.append(vector)
            # linalg = linear (линейная) + algebra (алгебра), norm обозначает норму
            # Вычисление нормы изображения
            norms.append(linalg.norm(vector, 2))
        a, b = vectors
        a_norm, b_norm = norms
        # dot возвращает скалярное произведение, вычисляется для двумерных массивов (матриц)
        res = dot(a / a_norm, b / b_norm)
        return res

    def __get_area_text(self, ocr_result):
        """
        Получение текстового содержимого внутри области субтитров
        """
        box, text = ocr_result
        coordinates = get_coordinates(box)
        area_text = []
        for content, coordinate in zip(text, coordinates):
            if self.sub_area is not None:
                s_ymin = self.sub_area[0]
                s_ymax = self.sub_area[1]
                s_xmin = self.sub_area[2]
                s_xmax = self.sub_area[3]
                xmin = coordinate[0]
                xmax = coordinate[1]
                ymin = coordinate[2]
                ymax = coordinate[3]
                if s_xmin <= xmin and xmax <= s_xmax and s_ymin <= ymin and ymax <= s_ymax:
                    area_text.append(content[0])
        return area_text

    def _compare_ocr_result(self, result_cache, img1, img1_no, img2, img2_no):
        """
        Сравнение, совпадает ли текст области субтитров, предсказанный для двух изображений
        """
        if self.ocr is None:
            self.ocr = OcrRecogniser()
        if img1_no in result_cache:
            area_text1 = result_cache[img1_no]['text']
        else:
            dt_box, rec_res = self.ocr.predict(img1)
            area_text1 = "".join(self.__get_area_text((dt_box, rec_res)))
            result_cache[img1_no] = {'text': area_text1, 'dt_box': dt_box, 'rec_res': rec_res}

        if img2_no in result_cache:
            area_text2 = result_cache[img2_no]['text']
        else:
            dt_box, rec_res = self.ocr.predict(img2)
            area_text2 = "".join(self.__get_area_text((dt_box, rec_res)))
            result_cache[img2_no] = {'text': area_text2, 'dt_box': dt_box, 'rec_res': rec_res}
        delete_no_list = []
        for no in result_cache:
            if no < min(img1_no, img2_no) - 10:
                delete_no_list.append(no)
        for no in delete_no_list:
            del result_cache[no]
        if ratio(area_text1, area_text2) > config.THRESHOLD_TEXT_SIMILARITY:
            return True
        else:
            return False

    @staticmethod
    def __is_coordinate_similar(coordinate1, coordinate2):
        """
        Проверка, похожи ли две координаты. Если разница между xmin, xmax, ymin, ymax двух координатных точек находится в пределах допуска по пикселям,
        то считается, что эти две координатные точки похожи
        """
        return abs(coordinate1[0] - coordinate2[0]) < config.PIXEL_TOLERANCE_X and \
            abs(coordinate1[1] - coordinate2[1]) < config.PIXEL_TOLERANCE_X and \
            abs(coordinate1[2] - coordinate2[2]) < config.PIXEL_TOLERANCE_Y and \
            abs(coordinate1[3] - coordinate2[3]) < config.PIXEL_TOLERANCE_Y

    @staticmethod
    def __get_thum(image, size=(64, 64), greyscale=False):
        """
        Унифицированная обработка изображения
        """
        # Изменение размера изображения с использованием image, Image.Resampling.LANCZOS для высокого качества
        image = image.resize(size, Image.Resampling.LANCZOS)
        if greyscale:
            # Преобразование изображения в режим L, который является градациями серого, каждый пиксель представлен 8 битами
            image = image.convert('L')
        return image

    def __delete_frame_cache(self):
        if not config.DEBUG_NO_DELETE_CACHE:
            if len(os.listdir(self.frame_output_dir)) > 0:
                for i in os.listdir(self.frame_output_dir):
                    os.remove(os.path.join(self.frame_output_dir, i))

    def empty_cache(self):
        """
        Удаление всех временных файлов, созданных в процессе извлечения субтитров
        """
        if not config.DEBUG_NO_DELETE_CACHE:
            if os.path.exists(self.temp_output_dir):
                shutil.rmtree(self.temp_output_dir, True)

    def update_progress(self, ocr=None, frame_extract=None):
        """
        Обновление прогресса
        :param ocr прогресс OCR
        :param frame_extract прогресс извлечения кадров видео
        """
        if ocr is not None:
            self.progress_ocr = ocr
        if frame_extract is not None:
            self.progress_frame_extract = frame_extract
        self.progress_total = (self.progress_frame_extract + self.progress_ocr) / 2

    def start_subtitle_ocr_async(self):
        def get_ocr_progress():
            """
            Получение прогресса распознавания OCR
            """
            # Получение общего количества кадров видео
            total_frame_count = self.frame_count
            # Флаг вывода подсказки о начале поиска субтитров
            notify = True
            while True:
                current_frame_no = self.subtitle_ocr_progress_queue.get(block=True)
                if notify:
                    print(config.interface_config['Main']['StartFindSub'])
                    notify = False
                self.update_progress(
                    ocr=100 if current_frame_no == -1 else (current_frame_no / total_frame_count * 100))
                # print(f'recv total_ms:{total_ms}')
                if current_frame_no == -1:
                    return

        process, task_queue, progress_queue = subtitle_ocr.async_start(self.video_path,
                                                                       self.raw_subtitle_path,
                                                                       self.sub_area,
                                                                       options={'REC_CHAR_TYPE': config.REC_CHAR_TYPE,
                                                                                'DROP_SCORE': config.DROP_SCORE,
                                                                                'SUB_AREA_DEVIATION_RATE': config.SUB_AREA_DEVIATION_RATE,
                                                                                'DEBUG_OCR_LOSS': config.DEBUG_OCR_LOSS,
                                                                                }
                                                                       )
        self.subtitle_ocr_task_queue = task_queue
        self.subtitle_ocr_progress_queue = progress_queue
        # Запуск потока для обновления прогресса OCR
        Thread(target=get_ocr_progress, daemon=True).start()
        return process

    @staticmethod
    def srt2txt(srt_file):
        subs = pysrt.open(srt_file, encoding='utf-8')
        output_path = os.path.join(os.path.dirname(srt_file), Path(srt_file).stem + '.txt')
        print(output_path)
        with open(output_path, 'w') as f:
            for sub in subs:
                f.write(f'{sub.text}\n')


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    # Запрос у пользователя пути к видео
    video_path = input(f"{config.interface_config['Main']['InputVideo']}").strip()
    # Запрос у пользователя области субтитров
    try:
        y_min, y_max, x_min, x_max = map(int, input(
            f"{config.interface_config['Main']['ChooseSubArea']} (ymin ymax xmin xmax)：").split())
        subtitle_area = (y_min, y_max, x_min, x_max)
    except ValueError as e:
        subtitle_area = None
    # Создание объекта извлечения субтитров
    se = SubtitleExtractor(video_path, subtitle_area, gui_mode=False)
    # Начало извлечения субтитров
    se.run()
