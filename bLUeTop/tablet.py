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
from enum import Enum

class bTablet:

    valuator = Enum('valuator', ['PressureValuator', 'TangentialPressureValuator', \
                                 'TiltValuator', 'VTiltValuator', 'HTiltValuator', 'NoValuator'])

    __widthValuator, __satValuator, __alphaValuator = (valuator.PressureValuator, ) * 3
    __alphaValuator = valuator.PressureValuator


    @classmethod
    def getWidthValuator(cls):
        return cls.__widthValuator

    @classmethod
    def setWidthValuator(cls, v):
        cls.__widthValuator = v

    @classmethod
    def getSatValuator(cls):
        return cls.__satValuator

    @classmethod
    def setSatValuator(cls, v):
        cls.__satValuator = v

    @classmethod
    def getAlphaValuator(cls):
        return cls.__alphaValuator

    @classmethod
    def setAlphaValuator(cls, v):
        cls.__alphaValuator = v

    def pressureToWidth( p):
        return p  + 0.2