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
import ctypes
from itertools import product
import numpy as np

from PySide6 import QtCore
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap, QIcon, QMouseEvent, QImageReader
from PySide6.QtWidgets import QListWidget, QListWidgetItem, \
    QSlider, QLabel, QDockWidget, QStyle, QColorDialog, QPushButton, QSizePolicy, QComboBox, QSpinBox
from PySide6.QtCore import Qt, QObject, QRect

from bLUeCore.rollingStats import movingVariance
from bLUeGui.bLUeImage import QImageBuffer
from bLUeGui.baseSignal import baseSignal_No

from version import BLUE_VERSION


def compat(v, version):
    if BLUE_VERSION[:2] == 'V2' and version[:2] == 'V6':
        v = v.replace('\\x12shiboken6.Shiboken', '\\x13shiboken2.shiboken2')
        v = v.replace('PySide6', 'PySide2')
    elif BLUE_VERSION[:2] == 'V6' and version[:2] != 'V6':
        v = v.replace('shiboken2.shiboken2', 'shiboken6.Shiboken')
        v = v.replace('PySide2', 'PySide6')
    return v


def imagej_description_metadata(description):
    """
    Modified version of tifffile.imagej_description_metadata()
    Return metatata from ImageJ image description as dict.
    Raise ValueError if not a valid ImageJ description.
    @param description:
    @type description: str
    @return:
    @rtype: dict
    """

    def _bool(val):
        return {'true': True, 'false': False}[val.lower()]

    result = {}
    for line in description.splitlines():
        try:
            key, val = line.split('=', 1)  # stop at first match, so char '=' is allowed in tag text
        except ValueError:
            continue
        key = key.strip()
        val = val.strip()
        for dtype in (int, float, _bool):
            try:
                val = dtype(val)
                break
            except (ValueError, KeyError):
                pass
        result[key] = val

    if 'ImageJ' not in result:
        raise ValueError('not an ImageJ image description')
    return result


def shift2D(arr, tr, fill=0):
    """
    Shift the two first axes of an array.
    The missing region is padded with the value fill
    (default 0). To be compliant with opencv images
    the first axis of arr is shifted by the second component of tr.
    The original array is not modified.
    @param arr: array
    @type arr: ndarray, ndims >= 2
    @param tr: 2-uple of translations for the 2 first axes
    @type tr: array-like
    @param fill: filling value
    @type fill: float
    @return: the shifted array
    @rtype: ndarray same shape and dtype as arr
    """
    s = arr.shape
    result = np.full(s, fill, dtype=arr.dtype)
    r1 = QRect(0, 0, s[1], s[0])
    r2 = QRect(tr[1], tr[0], s[1], s[0])
    r3 = QRect(-tr[1], -tr[0], s[1], s[0])
    r4, r5 = r1 & r2, r1 & r3
    if r4.isValid() and r5.isValid():
        result[r4.top():r4.bottom(), r4.left():r4.right()] = arr[r5.top():r5.bottom(), r5.left():r5.right()]
    return result


def array2DSlices(a2D, rect):
    """
    Return the 2-uple of slice objects convenient to
    index the intersection of the 2 dimensional array a2D
    with rect.
    @param a2D:
    @type a2D: ndarray, ndims >=2
    @param rect: (x, y, w, h)
    @type rect: 4-uple of int or QRect object
    @return:
    @rtype: 2-uple of slice objects
    """
    # convert rect to a QRect object
    if type(rect) not in [QRect]:
        try:
            rect = QRect(*rect)
        except (TypeError, ValueError):
            rect = QRect()
    # intersect a2D with rect
    qrect = QRect(0, 0, a2D.shape[1], a2D.shape[0]) & rect
    if qrect.isValid():
        return slice(qrect.top(), qrect.bottom()), slice(qrect.left(), qrect.right())
    else:
        return slice(0, 0), slice(0, 0)


def qColorToRGB(color):
    """
    Convert a QColor to its R,G,B components (range 0..255)
    @param color:
    @type color: QColor
    @return:
    @rtype: 3-uple of int
    """
    return color.red(), color.green(), color.blue()


def qColorToCMYK(color):
    """
    Converts a QColor to its C, M, Y, K components (range 0..255)
    @param color:
    @type color: QColor
    @return:
    @rtype: 4-uple of int
    """
    return color.cyan(), color.magenta(), color.yellow(), color.black()


def qColorToHSV(color):
    """
    Converts a QColor to its H,S,V components
    @param color:
    @type color: QColor
    @return:
    @rtype: 3-uple of int
    """
    return color.hue(), color.saturation(), color.value()


class colorInfoView(QDockWidget):
    """
    Display formatted color info for a pixel
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.label = QLabel()
        self.label.setWindowTitle('Info')
        self.setWidget(self.label)
        self.setWindowTitle(self.label.windowTitle())
        self.setFocusPolicy(Qt.ClickFocus)
        self.label.setStyleSheet("font-family: 'courier'; font-size: 8pt")
        self.setWhatsThis(
            """<b>Info</b><br>
            Input/output pixel values for the active layer.<br>
            Values are displayed in the RGB, CMYK and HSV color spaces.
            For each space, inputs are shown in the left column
            and outputs in the right column.<br>
            The layer mask is ignored.<br>
            """
        )  # end of setWhatsThis

    def setText(self, clrI, clrC):
        """
        Set widget text to formatted color info
        @param clrI: input color
        @type clrI: QColor
        @param clrC: output color
        @type clrC: QColor
        """
        r0 = 'R ' + "".join([str(w).ljust(4) if type(w) is int else w
                             for w in (clrI.red(), clrC.red(), '|C ', clrI.cyan() * 100 // 255,
                                       clrC.cyan() * 100 // 255, '|H ', clrI.hue(), clrC.hue())])
        r1 = 'G ' + "".join([str(w).ljust(4) if type(w) is int else w
                             for w in (clrI.green(), clrC.green(), '|M ',
                                       clrI.magenta() * 100 // 255, clrC.magenta() * 100 // 255, '|S ',
                                       clrI.saturation() * 100 // 255, clrC.saturation() * 100 // 255)])
        r2 = 'B ' + "".join([str(w).ljust(4) if type(w) is int else w
                             for w in (clrI.blue(), clrC.blue(), '|Y ',
                                       clrI.yellow() * 100 // 255, clrC.yellow() * 100 // 255, '|V ',
                                       clrI.value() * 100 // 255, clrC.value() * 100 // 255)])
        r3 = "".join((' ',) * 10) + '|K ' + "".join([str(w).ljust(4) for w in
                                                     (clrI.black() * 100 // 255, clrC.black() * 100 // 255)])
        self.label.setText('\n'.join((r0, r1, r2, r3)))


def hideConsole():
    """
    Hides the console window
    """
    whnd = ctypes.windll.kernel32.GetConsoleWindow()
    if whnd != 0:
        ctypes.windll.user32.ShowWindow(whnd, 0)
        ctypes.windll.kernel32.CloseHandle(whnd)


def showConsole():
    """
    Shows the console window
    """
    whnd = ctypes.windll.kernel32.GetConsoleWindow()
    if whnd != 0:
        ctypes.windll.user32.ShowWindow(whnd, 1)
        ctypes.windll.kernel32.CloseHandle(whnd)


def multiply(matr_a, matr_b):
    """Return product of an MxP matrix A with an PxN matrix B."""
    cols, rows = len(matr_b[0]), len(matr_b)
    resRows = range(len(matr_a))
    rMatrix = [[0] * cols for _ in resRows]
    for idx in resRows:
        for j, k in product(range(cols), range(rows)):
            rMatrix[idx][j] += matr_a[idx][k] * matr_b[k][j]
    return rMatrix


def inversion(m):
    """
    @param m:
    @type m:
    @return:
    @rtype:
    """
    m1, m2, m3, m4, m5, m6, m7, m8, m9 = m.ravel()
    inv = np.array([[m5 * m9 - m6 * m8, m3 * m8 - m2 * m9, m2 * m6 - m3 * m5],
                    [m6 * m7 - m4 * m9, m1 * m9 - m3 * m7, m3 * m4 - m1 * m6],
                    [m4 * m8 - m5 * m7, m2 * m7 - m1 * m8, m1 * m5 - m2 * m4]])
    return inv / multiply(inv[0], m[:, 0])


class UDict(object):
    """
    Union of dictionaries. The dictionaries are neither copied nor changed.
    Examples :  UDict(()), UDict((d1,)), UDict((d1,d2))
    """

    def __init__(self, *args):
        """
        If args is a tuple of dict instances, build an (ordered) union
        of the dictionaries : __getitem__(key) returns the first found
        value corresponding to the key, and None if the key is not present
        in any of the dictionaries. No exception is raised if the key does not
        exist.
        @param args: tuple of dict
        @type args:
        """
        if args:
            self.__dictionaries = args[0]  # tuple of dict
        else:
            self.__dictionaries = ()

    def __getitem__(self, item):
        for i in range(len(self.__dictionaries)):
            if item in self.__dictionaries[i]:
                return self.__dictionaries[i][item]
        return None

    @property
    def dictionaries(self):
        """

        @return:
        @rtype: tuple of dictionaries
        """
        return self.__dictionaries


class QbLUeColorDialog(QColorDialog):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.closeSignal = baseSignal_No()

    def closeEvent(self, e):
        self.closeSignal.sig.emit()
        # remove possible links to an adjust form
        try:
            self.currentColorChanged.disconnect()
            self.colorSelected.disconnect()
        except RuntimeError:
            pass


class QbLUeComboBox(QComboBox):

    def __getstate__(self):
        return {'text': self.currentText()}

    def __setstate__(self, state):
        ind = self.findText(state['text'])
        if ind != -1:
            self.setCurrentIndex(ind)


class QbLUeSpinBox(QSpinBox):

    def __getstate__(self):
        return {'value': self.value()}

    def __setstate__(self, state):
        self.setValue(state['value'])


class QbLUeSlider(QSlider):
    """
    Enhanced QSlider.
    Override mousepressevent to prevent jumps
    when clicking the handle and to update value
    with a single jump when clicking on the groove.
    """
    bLueSliderDefaultColorStylesheet = """QSlider::groove:horizontal:enabled { 
                                                                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 blue, stop:1 red);
                                                                }
                                                                QSlider::groove:horizontal:disabled {
                                                                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8888FF, stop:1 #FF8888);
                                                                }"""
    bLueSliderDefaultMGColorStylesheet = """QSlider::groove:horizontal:enabled {
                                                                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 magenta, stop:1 green);
                                                                }
                                            QSlider::groove:horizontal:disabled {
                                                                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8888FF, stop:1 #FF8888);
                                                                }"""
    bLueSliderDefaultIMGColorStylesheet = """QSlider::groove:horizontal:enabled {
                                                                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 green, stop:1 magenta);
                                                                }
                                             QSlider::groove:horizontal:disabled {
                                                                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8888FF, stop:1 #FF8888);
                                                                }"""
    bLueSliderDefaultIColorStylesheet = """QSlider::groove:horizontal:enabled {
                                                                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 red, stop:1 blue);
                                                                }
                                            QSlider::groove:horizontal:disabled { 
                                                                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8888FF, stop:1 #FF8888);
                                                                }"""
    bLueSliderDefaultBWStylesheet = """QSlider::groove:horizontal:enabled {
                                                                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #333333, stop:1 white);
                                                                }
                                       QSlider::groove:horizontal:disabled {
                                                                background: #888888;
                                                                }"""
    bLueSliderDefaultIBWStylesheet = """QSlider::groove:horizontal:enabled {
                                                                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 gray, stop:1 #333333);
                                                                }
                                        QSlider::groove:horizontal:disabled {
                                                                background: #888888;
                                                                }"""

    def __init__(self, *args, **kwargs):
        super(QbLUeSlider, self).__init__(*args, **kwargs)
        self.setTickPosition(QSlider.NoTicks)
        self.setMaximumSize(16777215, 10)

    def __getstate__(self):
        return {'value': self.value()}

    def __setstate__(self, state):
        self.setValue(state['value'])

    def mousePressEvent(self, event):
        """
        Update the slider value with a single jump when clicking on the groove.
        # min is at left or top. Change upsideDown to True to reverse this behavior.
        @param event:
        @type event:
        """
        pressVal = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), event.x(), self.width(),
                                                  upsideDown=False)
        if abs(pressVal - self.value()) > (
                self.maximum() - self.minimum()) * 20 / self.width():  # handle width should be near 20
            self.setValue(pressVal)
        else:
            super().mousePressEvent(event)


class QbLUeLabel(QLabel):
    """
    Emits a signal when double clicked
    """
    doubleClicked = QtCore.Signal()

    def mouseDoubleClickEvent(self, e):
        self.doubleClicked.emit()


class QbLUePushButton(QPushButton):
    """
    Form PushButtons (specific style sheet)
    """
    pass


class historyList(list):
    """
    History management.
    Successive states are saved into a stack,
    implemented by a list, with top of stack
    at index 0.
    The stack alaways contains the size latest states.
    No copy is done, so objects should be copied
    before and after calling addItem, undo and redo.
    """

    def __init__(self, size=5):
        """
        self.current is the list index of the last restored state, -1 if
        no state was restored since the last call to addItem.
        @param size: max history size
        @type size: int
        """
        super().__init__()
        self.size = size
        self.current = -1

    def addItem(self, item):
        """
        Pushes an item on the stack.
        @param item:
        @type item: any type
        """
        if self.current == - 1:
            super().insert(0, item)
        if len(self) > self.size:
            self.pop()
        self.current = -1

    def undo(self, saveitem=None):
        """
        Returns the previous item from history.
        The Parameter saveitem is used to save the current state before undo
        (i.e. before restoration), if it is not already a restored state.
        The method raises ValueError if there is no more item to restore.
        @param saveitem:
        @type saveitem: object
        @return:
        @rtype: object
        """
        if self.current >= len(self) - 1:
            # no more items to restore
            raise ValueError
        # stack a possibly unsaved (i.e. not restored) item
        if (self.current == -1) and (saveitem is not None):
            self.addItem(saveitem)
            self.current = 0
        self.current += 1
        item = self[self.current]
        return item

    def redo(self):
        """
        Returns the next item in stack, if any.
        Otherwise, raises ValueError.
        @return:
        @rtype: object
        """
        if self.current <= 0:
            raise ValueError
        self.current -= 1
        return self[self.current]

    def canUndo(self):
        return self.current < len(self) - 1

    def canRedo(self):
        return self.current > 0


class optionsWidgetItem(QListWidgetItem):
    def __init__(self, *args, intName='', **kwargs, ):
        super().__init__(*args, **kwargs)
        self._internalName = intName

    @property
    def internalName(self):
        return self._internalName

    def isChecked(self):
        return self.checkState() == Qt.CheckState.Checked


class bLUeEventFilter(QObject):

    def eventFilter(self, target, e):
        """
        Filter mouse events for disabled items
        @param target:
        @type target:
        @param e:
        @type e:
        """
        if isinstance(e, QMouseEvent):
            return not (target.flags() & Qt.itemIsEnabled)
        return False


class optionsWidget(QListWidget):
    """
    Display a list of options with checkboxes.
    The choices can be mutually exclusive (default) or not
    exclusive. Actions can be done on item selection by assigning
    a function to onSelect. It is called after the selection of the new item.
    Passing a signal or function to the parameter changed enables to trigger an action when,
    and only when, clicking an item induces a change in checkbox states.
    """
    # ad hoc signal triggered when item clicked AND change in checkbox states (see method select)
    userCheckStateChanged = QtCore.Signal(QListWidgetItem)

    def __init__(self, options=None, optionNames=None, exclusive=True, changed=None, parent=None,
                 flow=QListWidget.TopToBottom):
        """
        @param options: list of options
        @type options: list of str
        @param optionNames: list of displayed names corresponding to options
        @type optionNames: list of str
        @param exclusive:
        @type exclusive: bool
        @param changed: signal or slot for itemclicked signal
        @type changed: signal or function (0 or 1 parameter of type QListWidgetItem)
        @param parent:
        @type parent: QObject
        @param flow:  which direction the items layout should flow
        @type flow: QListView.Flow
        """
        super().__init__(parent)

        if flow is not None:
            self.setFlow(flow)
        if options is None:
            options = []
        if optionNames is None:
            self.extNames = options
        else:
            self.extNames = optionNames
        self.intNames = options
        # dict of items with internal names of options as keys,
        # unfortunately shadowing a QListWidget built-in method
        self.items = {}
        # dict of item states (True, False) with option internal name as key
        self.options = {}
        self.changed = changed
        for intName, name in zip(self.intNames, self.extNames):
            listItem = optionsWidgetItem(name, self, intName=intName)
            listItem.setCheckState(Qt.Unchecked)
            self.addItem(listItem)
            self.items[intName] = listItem
            self.options[intName] = (listItem.checkState() == Qt.Checked)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        if flow == QListWidget.TopToBottom:
            self.setMinimumHeight(self.sizeHintForRow(0) * len(options))
            self.setMaximumHeight(self.sizeHintForRow(0) * len(options) + 10)
        else:  # QListWidget.LeftToRight
            self.setMinimumWidth(self.sizeHintForColumn(0) * len(options))
            self.setMaximumWidth(self.sizeHintForColumn(0) * len(options) + 10)
        self.exclusive = exclusive
        self.itemClicked.connect(self.select)
        self.userCheckStateChanged.connect(self.changed)
        # selection hook.
        self.onSelect = lambda x: 0

    def __getstate__(self):
        """
        returns a pickable dict capturing instance state
        @return:
        @rtype: dict
        """
        return dict([(it, self.items[it].checkState()) for it in self.items])

    def __setstate__(self, state):
        """

        @param state:
        @type state: dict
        """
        for itemName in state:
            self.items[itemName].setCheckState(state[itemName])
            if state[itemName] == Qt.Checked:
                self.select(self.items[itemName])

    def select(self, item, callOnSelect=True):
        """
        Item clicked slot. It updates the state of the items and
        the dict of options. Next, if callOnSelect is True, onSelect is called.
        Finally, if an item was modified by a mouse click, then
        self.changed is called/emitted.
        @param item:
        @type item: QListWidgetItem
        @param callOnSelect:
        @type callOnSelect: bool
        """
        # don't react to mouse click on disabled items
        if not (item.flags() & Qt.ItemIsEnabled):
            return
        # Update item states:
        # if exclusive, clicking on an item should turn it
        # into (or keep it) checked. Otherwise, there is nothing to do
        # because select is called after the item state has changed.
        if self.exclusive:
            for r in range(self.count()):
                currentItem = self.item(r)
                if currentItem is not item:
                    currentItem.setCheckState(Qt.Unchecked)
                else:
                    currentItem.setCheckState(Qt.Checked)
        # update options dict
        modified = False
        for option in self.options.keys():
            newState = self.items[option].checkState() == Qt.Checked
            if self.options[option] != newState:
                self.options[option] = newState
                modified = True
        if callOnSelect and modified:
            self.onSelect(item)
        if modified and self.sender() is not None:
            # item clicked and checkbox state modified
            self.userCheckStateChanged.emit(item)

    def checkOption(self, name, checked=True, callOnSelect=True):
        """
        Check or (for non exclusive options only) uncheck an item.
        Next, if callOnSelect is True, onSelect is called.
        A ValueError exception is raised  if an attempt is done to
        uncheck an item in a list of mutually exclusive options.
        @param name: internal name of option
        @type name: str
        @param checked: check/uncheck flag
        @type checked: bool
        @param callOnSelect:
        @type callOnSelect: bool
        """
        item = self.items[name]
        if not checked and self.exclusive:
            raise ValueError('For mutually exclusive options, unchecking is not possible. Please check another item')
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.select(item, callOnSelect=callOnSelect)

    def unCheckAll(self):
        if self.exclusive:
            return
        for r in range(self.count()):
            self.item(r).setCheckState(Qt.Unchecked)

    def checkAll(self):
        if self.exclusive:
            return
        for r in range(self.count()):
            self.item(r).setCheckState(Qt.Checked)

    @property
    def checkedItems(self):
        return [self.item(i) for i in range(self.count()) if self.item(i).checkState() == Qt.Checked]


def checkeredImage(format=QImage.Format_ARGB32):
    """
    Returns a 20x20 checker
    @param format:
    @type format:
    @return: checker
    @rtype: QImage
    """
    base = QImage(20, 20, format)
    qp = QPainter(base)
    qp.setCompositionMode(QPainter.CompositionMode_Source)
    qp.fillRect(0, 0, 10, 10, Qt.gray)
    qp.fillRect(10, 0, 10, 10, Qt.white)
    qp.fillRect(0, 10, 10, 10, Qt.white)
    qp.fillRect(10, 10, 10, 10, Qt.gray)
    qp.end()
    return base


class stateAwareQDockWidget(QDockWidget):
    """
    Record the current tabbing state.
    Needed to restore the workspace when switching from a document to another one.
    This attribute should be restored if the change does not result from a user
    drag and drop action (see layerView.closeAdjustForms for an example)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFloating(True)  # default :  left docking area
        self.tabbed = False
        self._closed = False

        def f(b):
            self.tabbed = not b

        self.topLevelChanged.connect(f)

    def closeEvent(self, event):
        self._closed = True
        super().closeEvent(event)

    @property
    def isClosed(self):
        return self._closed


def clip(image, mask, inverted=False):
    """
    clip an image by applying a mask to its alpha channel
    @param image:
    @type image:
    @param mask:
    @type mask:
    @param inverted:
    @type inverted:
    @return:
    @rtype:
    """
    bufImg = QImageBuffer(image)
    bufMask = QImageBuffer(mask)
    if inverted:
        bufMask = bufMask.copy()
        bufMask[:, :, 3] = 255 - bufMask[:, :, 3]
    bufImg[:, :, 3] = bufMask[:, :, 3]


def QImageFromFile(filename):
    """

    @param filename:
    @type filename: str
    @return:
    @rtype: QImage
    """
    reader = QImageReader(filename)
    reader.setAutoTransform(True)  # handle orientation tag
    return reader.read()


if __name__ == '__main__':
    a = np.ones(100, dtype=int).reshape(10, 10)
    # b=strides_2d(a, (11,11))
    m = movingVariance(a, 7)
    print(m)
