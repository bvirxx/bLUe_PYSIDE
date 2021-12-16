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

##################
# Signal classes.
# They are signal containers:
# they can be used as a substitute to
# multiple inheritance for adding custom
# signals to classes that do not inherit from
# QObject (cf. QLayer).
##################

from PySide6 import QtCore
from PySide6.QtCore import QObject


class baseSignal_No(QObject):
    sig = QtCore.Signal()


class baseSignal_bool(QObject):
    sig = QtCore.Signal(bool)


class baseSignal_Int2(QObject):
    sig = QtCore.Signal(int, int, QtCore.Qt.KeyboardModifiers)
