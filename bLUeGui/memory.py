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
import weakref


def weakProxy(ref):
    """
    Return a proxy weak reference to a python object.
    It can be used as the original object, without increasing
    its reference count.
    @param ref: reference to a python object
    @type ref: object
    @return: weak reference to object
    @rtype: weakref
    """
    if ref is None:
        return ref
    if type(ref) in weakref.ProxyTypes:
        return ref
    else:
        return weakref.proxy(ref)