"""
This File is part of bLUe software.

Copyright (C) 2017  Bernard Virot <bernard.virot@libertysurf.fr>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Lesser Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler

def initLogging():

    formatter = logging.Formatter("%(process)d: %(message)s")

    file_handler = RotatingFileHandler('log.txt',
                                       mode='a',
                                       maxBytes=5*1024*1024,
                                       backupCount=2,
                                       encoding=None,
                                       delay=False
                                       )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger('blue')
    logger.setLevel('INFO')
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def excHandler(exc_type, exc_value, exc_traceback):
    logger.error('PID %d' % os.getpid(), exc_info=(exc_type, exc_value, exc_traceback))
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)


logger = initLogging()

sys.excepthook = excHandler
