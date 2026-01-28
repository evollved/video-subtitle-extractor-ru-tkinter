#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный файл запуска графического интерфейса
"""
import sys
import os

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(__file__))

# Запускаем GUI
from gui_tkinter import SubtitleExtractorGUI

if __name__ == '__main__':
    app = SubtitleExtractorGUI()
    app.run()