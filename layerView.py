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

from PySide.QtCore import QRectF
from PySide.QtGui import QAction, QMenu, QSlider, QImage, QStyle, QPalette, QColor, QListWidget, QCheckBox, QMessageBox
from PySide.QtGui import QBrush
from PySide.QtGui import QComboBox
from PySide.QtGui import QFontMetrics
from PySide.QtGui import QHBoxLayout
from PySide.QtGui import QLabel
from PySide.QtGui import QPainter
from PySide.QtGui import QTableView, QStandardItem, QStandardItemModel, QAbstractItemView, QStyledItemDelegate, QPixmap, QIcon, QHeaderView
from PySide.QtCore import Qt
from PySide.QtGui import QTextOption
from PySide.QtGui import QVBoxLayout

import resources_rc  # mandatory : DO NOT REMOVE !!!
import QtGui1



class layerModel(QStandardItemModel):

    def __init__(self):
        super(layerModel, self).__init__()

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled | Qt.ItemIsEditable



class itemDelegate(QStyledItemDelegate):
    """
    Item painting

    """
    def __init__(self, parent=None):
        QStyledItemDelegate.__init__(self, parent)

    def paint(self, painter, option, index):
        rect = QRectF(option.rect)
        # mask column
        if index.column() == 2:
            if self.parent().img is not None:
                if self.parent().img.layersStack[-1 - index.row()].maskIsSelected:
                    text = 'M *'
                else:
                    text = 'M  '
                if self.parent().img.layersStack[-1 - index.row()].maskIsEnabled:
                    painter.save()
                    painter.setPen(Qt.red)
                    painter.drawText(rect, text, QTextOption(Qt.AlignCenter))
                    painter.restore()
                    return
                painter.drawText(rect, text, QTextOption(Qt.AlignCenter))
        elif index.column() == 0:
            painter.save()
            #painter.setPen(Qt.red)
            if option.state & QStyle.State_Selected:
                c = option.palette.color(QPalette.Highlight)
                painter.fillRect(rect, c)
            if self.parent().img.layersStack[-1 - index.row()].visible:
                px = self.inv_px1 if option.state & QStyle.State_Selected else self.px1
            else:
                px = self.inv_px2 if option.state & QStyle.State_Selected else self.px2
            painter.drawPixmap(rect, px, QRectF(0,0,self.px1.width(), self.px1.height()))
            painter.restore()
        else:
            # call default
            QStyledItemDelegate.paint(self, painter, option, index)

class QLayerView(QTableView) :
    """
    The class QLayerView inherits from QTableView. It is used
    in the main form to display the stack of image layers.
    """
    def __init__(self, parent):
        super(QLayerView, self).__init__(parent)
        self.img = None
        # form to display
        self.currentWin = None
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.clicked.connect(self.viewClicked)
        self.customContextMenuRequested.connect(self.contextMenu)
        # behavior and style for selection
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        delegate = itemDelegate(parent=self)
        self.setItemDelegate(delegate)
        ic1 = QImage(":/images/resources/eye-icon.png")
        ic2 = QImage(":/images/resources/eye-icon-strike.png")
        delegate.px1 = QPixmap.fromImage(ic1)
        delegate.px2 = QPixmap.fromImage(ic2)
        ic1.invertPixels()
        ic2.invertPixels()
        delegate.inv_px1 = QPixmap.fromImage(ic1)
        delegate.inv_px2 = QPixmap.fromImage(ic2)


        """
        self.verticalHeader().setMovable(True)
        self.verticalHeader().setDragEnabled(True)
        self.verticalHeader().setDragDropMode(QAbstractItemView.InternalMove)
        """
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDragDropOverwriteMode(False)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)

        # verticallayout
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignBottom)

        # Preview option
        # We should use a QListWidget or a custom optionsWidget
        # (cf. utils.py) :  adding it to QVBoxLayout with mode
        # Qt.AlignBottom does not work.
        # TODO : try setSizePolicy(QSizePolicy.Fixed , QSizePolicy.Fixed)
        self.previewOptionBox = QCheckBox('Preview')
        self.previewOptionBox.setMaximumSize(100, 30)
        self.previewOptionBox.setChecked(False)
        l.addWidget(self.previewOptionBox)
        # state changed event handler
        def m(state): # Qt.Checked Qt.UnChecked
            if self.img is None:
                return
            if state == Qt.Checked:
                self.img.useThumb = True
            else:
                self.img.useThumb = False
            QtGui1.window.updateStatus()
            self.img.cacheInvalidate()
            if not self.img.useThumb:
                info = QMessageBox()
                info.setWindowModality(Qt.ApplicationModal)
                info.setWindowTitle('Information')
                info.setIcon(QMessageBox.Information)
                info.setText('Updating all layers.....Please wait')
                info.show()
            QtGui1.app.processEvents()
            self.img.layersStack[0].applyToStack()
            QtGui1.window.label.repaint()
            QtGui1.app.processEvents()

        self.previewOptionBox.stateChanged.connect(m)

        # opcity slider
        self.wdgt = QSlider(Qt.Horizontal)
        self.wdgt.setTickPosition(QSlider.TicksBelow)
        self.wdgt.setRange(0, 100)
        self.wdgt.setSingleStep(1)
        self.wdgt.setSliderPosition(100)
        opacityLabel = QLabel()
        opacityLabel.setMaximumSize(100,30)
        #self.opacityLabel.setStyleSheet("QLabel {background-color: white;}")
        opacityLabel.setText("Layer opacity")
        l.addWidget(opacityLabel)

        hl =  QHBoxLayout()
        self.opacityValue = QLabel()
        font = self.opacityValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("100 ")
        h = metrics.height()
        self.opacityValue.setMinimumSize(w, h)
        self.opacityValue.setMaximumSize(w, h)

        self.opacityValue.setText('100 ')
        self.opacityValue.setStyleSheet("QLabel {background-color: white;}")
        hl.addWidget(self.opacityValue)
        hl.addWidget(self.wdgt)
        l.addLayout(hl)
        l.setContentsMargins(20,0,20,25) # left, top, right, bottom
        self.setLayout(l)

        # opacity value changed event handler
        def f():
            self.opacityValue.setText(str('%d ' % self.wdgt.value()))
            self.img.getActiveLayer().setOpacity(self.wdgt.value())
            self.img.onImageChanged()

        self.wdgt.valueChanged.connect(f)

        # blending modes combo box
        compLabel = QLabel()
        compLabel.setText("Composition Mode")
        l.addWidget(compLabel)
        modes = ['Normal', 'Plus', 'Multiply', 'Screen', 'Overlay', 'Darken', 'Lighten', 'Color Dodge', 'Color Burn', 'Hard Light',
                'Soft Light', 'Difference', 'Exclusion']

        self.compositionModeDict = { 'Normal':QPainter.CompositionMode_SourceOver,
                                'Plus':QPainter.CompositionMode_Plus, 'Multiply':QPainter.CompositionMode_Multiply,
                                'Screen':QPainter.CompositionMode_Screen, 'Overlay':QPainter.CompositionMode_Overlay,
                                'Darken':QPainter.CompositionMode_Darken, 'Lighten':QPainter.CompositionMode_Lighten,
                                'Color Dodge':QPainter.CompositionMode_ColorDodge, 'Color Burn':QPainter.CompositionMode_ColorBurn,
                                'Hard Light':QPainter.CompositionMode_HardLight, 'Soft Light':QPainter.CompositionMode_SoftLight,
                                'Difference':QPainter.CompositionMode_Difference, 'Exclusion':QPainter.CompositionMode_Exclusion
                                }

        self.blendingModeCombo = QComboBox()
        l.addWidget(self.blendingModeCombo)
        self.blendingModeCombo.addItems(modes)

        def g(ind):
            s = self.blendingModeCombo.currentText()
            self.img.getActiveLayer().compositionMode = self.compositionModeDict[str(s)]
            self.img.onImageChanged()

        self.blendingModeCombo.currentIndexChanged.connect(g)

    def closeAdjustForms(self):
        if self.img is None:
            return
        stack = self.img.layersStack
        for i in xrange(len(stack)):
            if hasattr(stack[i], "view"):
                if stack[i].view is not None:
                    stack[i].view.close()

    def setLayers(self, mImg):
        """
        sets img attribute to mImg and shows the stack of layers from mImg
        :param mImg: mImage
        """
        self.closeAdjustForms()
        self.img=mImg
        mImg.layerView = self
        model = layerModel()
        model.setColumnCount(3)

        #self.setModel(model)
        l = len(mImg.layersStack)

        for r, lay in enumerate(reversed(mImg.layersStack)):
            items = []
            # col 0 : visibility icon
            if lay.visible :
                item_visible = QStandardItem(QIcon(":/images/resources/eye-icon.png"), "")
            else:
                item_visible = QStandardItem(QIcon(":/images/resources/eye-icon-strike.png"), "")
            items.append(item_visible)
            # col 1 : image icon and name
            smallImg = lay.resize(200)
            item_name = QStandardItem(QIcon(QPixmap.fromImage(smallImg)), lay.name)
            items.append(item_name)
            item_mask = QStandardItem('M')
            items.append(item_mask)
            model.appendRow(items)

        #model.setData(model.index(0, 2), QBrush(Qt.red), Qt.ForegroundRole | Qt.DecorationRole)

        self.setModel(model)

        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        header = self.horizontalHeader()
        header.setResizeMode(0, QHeaderView.ResizeToContents)
        header.setResizeMode(1, QHeaderView.ResizeToContents)
        header.setResizeMode(2, QHeaderView.ResizeToContents)
        # select active layer
        self.selectRow(len(mImg.layersStack) - 1 - mImg.activeLayerIndex)

        self.update()
        """
        activeLayer = mImg.getActiveLayer()

        self.currentWin = activeLayer.view

        if hasattr(activeLayer, 'adjustView'):
            self.currentWin = activeLayer.adjustView
        if hasattr(activeLayer, 'segmentView'):
            self.currentWin = activeLayer.segmentView
        if self.currentWin is not None:
            self.currentWin.show()
        """
    def update(self):
        activeLayer = self.img.getActiveLayer()
        if hasattr(activeLayer, 'view'):
            self.currentWin = activeLayer.view
        if self.currentWin is not None:
            self.currentWin.show()
            self.currentWin.activateWindow()

    def dropEvent(self, event):
        """
        drop event handler : moving layer
        :param event:
        """
        if event.source() == self:
            rows = set([mi.row() for mi in self.selectedIndexes()])
            targetRow = self.indexAt(event.pos()).row()
            rows.discard(targetRow)
            rows = sorted(rows)
            if not rows:
                return
            if targetRow == -1:
                targetRow = self.model().rowCount()
            # insert empty row
            for _ in range(len(rows)):
                self.model().insertRow(targetRow)
            # src row to target row mapping
            rowMapping = dict()
            for idx, row in enumerate(rows):
                if row < targetRow:
                    rowMapping[row] = targetRow + idx
                else:
                    rowMapping[row + len(rows)] = targetRow + idx
            colCount = self.model().columnCount()
            for srcRow, tgtRow in sorted(rowMapping.iteritems()):
                for col in range(0, colCount):
                    self.model().setItem(tgtRow, col, self.model().takeItem(srcRow, col))
            for row in reversed(sorted(rowMapping.iterkeys())):
                self.model().removeRow(row)

            rStack = self.img.layersStack[::-1]
            for _ in range(len(rows)):
                rStack.insert(targetRow, None)
            for srcRow, tgtRow in sorted(rowMapping.iteritems()):
                rStack[tgtRow] = rStack[srcRow]
            for row in reversed(sorted(rowMapping.iterkeys())):
                rStack.pop(row)

            self.img.layersStack = rStack[::-1]
            #event.accept()

    def viewClicked(self, clickedIndex):
        """
        Mouse click event handler.
        :param clickedIndex:
        """
        row = clickedIndex.row()
        # toggle layer visibility
        if clickedIndex.column() == 0 :
            # background layer is always visible
            if row == len(self.img.layersStack) - 1:
                return
            visible = not(self.img.layersStack[-1-row].visible)
            self.img.layersStack[-1-row].visible = visible
            # update visibility icon
            if visible:
                self.model().setData(clickedIndex, QIcon(":/images/resources/eye-icon.png") ,Qt.DecorationRole)
            else:
                self.model().setData(clickedIndex, QIcon(":/images/resources/eye-icon-strike.png"), Qt.DecorationRole)
        # hide/display adjustment form
        elif clickedIndex.column() == 1 :
            # make selected layer the active layer
            self.img.setActiveLayer(len(self.img.layersStack) - 1 - row, signaling=False)
            # update displayed window
            if self.currentWin is not None:
                self.currentWin.hide()
                self.currentWin=None
            if hasattr(self.img.layersStack[-1-row], "view"):
                if self.img.layersStack[-1-row].view is not None:
                    self.currentWin = self.img.layersStack[-1-row].view
            if hasattr(self.img.layersStack[-1-row], "view"):
                if self.img.layersStack[-1-row].view is not None:
                    self.currentWin = self.img.layersStack[-1 - row].view
            if self.currentWin is not None:
                self.currentWin.show()
                self.currentWin.activateWindow()
        # select mask
        elif clickedIndex.column() == 2:
            self.img.layersStack[-1-clickedIndex.row()].maskIsSelected = not self.img.layersStack[-1-clickedIndex.row()].maskIsSelected
            #self.repaint()

    def contextMenu(self, pos):
        """
        context menu event handler
        :param pos: event coordinates (relative to widget)
        """
        index = self.indexAt(pos)
        layer = self.img.layersStack[-1-index.row()]
        menu = QMenu()
        actionTransparency = QAction('Transparency', None)
        actionDup = QAction('Duplicate layer', None)
        actionMaskEnable = QAction('Enable mask', None)
        actionMaskDisable = QAction('Disable mask', None)
        actionMaskReset = QAction('Reset mask', None)
        menu.addAction(actionTransparency)
        menu.addAction(actionDup)
        menu.addAction(actionMaskEnable)
        menu.addAction(actionMaskDisable)
        menu.addAction(actionMaskReset)

        def f():
            self.wdgt.show()
        def g(value):
            layer.setOpacity(value)
        def dup():
            self.img.dupLayer(index = len(self.img.layersStack) -1 - index.row())
            self.setLayers(self.img)
        def maskEnable():
            layer.maskIsEnabled = True
            layer.updatePixmap()
        def maskDisable():
            layer.maskIsEnabled = False
            layer.updatePixmap()
        def maskReset():
            layer.resetMask()
            layer.updatePixmap()
        actionTransparency.triggered.connect(f)
        actionDup.triggered.connect(dup)
        actionMaskEnable.triggered.connect(maskEnable)
        actionMaskDisable.triggered.connect(maskDisable)
        actionMaskReset.triggered.connect(maskReset)
        self.wdgt.valueChanged.connect(g)


        menu.exec_(self.mapToGlobal(pos))


