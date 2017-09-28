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
import cv2
import numpy as np
from PySide2.QtCore import QRectF, QSize, Qt
from PySide2.QtGui import QImage, QPalette, QColor, QKeySequence, QFontMetrics, QTextOption, QPixmap, QIcon, QPainter, QStandardItem, QStandardItemModel
from PySide2.QtWidgets import QAction, QMenu, QSlider, QStyle, QListWidget, QCheckBox, QMessageBox, QApplication
from PySide2.QtWidgets import QComboBox, QHBoxLayout, QLabel, QTableView, QAbstractItemView, QStyledItemDelegate, QHeaderView, QVBoxLayout

import resources_rc  # mandatory : DO NOT REMOVE !!!
import QtGui1
from imgconvert import QImageBuffer

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
        # context menu
        #self.setContextMenuPolicy(Qt.CustomContextMenu)
        # mouse click
        self.clicked.connect(self.viewClicked)
        #self.customContextMenuRequested.connect(self.contextMenu)
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
        self.setIconSize(QSize(20,15))

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
        self.previewOptionBox.setChecked(True)
        l.addWidget(self.previewOptionBox)
        # state changed event handler
        def m(state): # Qt.Checked Qt.UnChecked
            if self.img is None:
                return
            self.img.useThumb = (state == Qt.Checked)
            QtGui1.window.updateStatus()
            self.img.cacheInvalidate()
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                """
                info = QMessageBox()
                info.setWindowModality(Qt.ApplicationModal)
                info.setWindowTitle('Information')
                info.setIcon(QMessageBox.Information)
                info.setText('Updating all layers.....Please wait')
                info.show()
                """
                QtGui1.app.processEvents()
                # update stack
                self.img.layersStack[0].applyToStack()
            finally:
                QApplication.restoreOverrideCursor()
            QtGui1.window.label.repaint()

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
        for key in modes:
            self.blendingModeCombo.addItem(key, self.compositionModeDict[key])
        def g(ind):
            s = self.blendingModeCombo.currentText()
            self.img.getActiveLayer().compositionMode = self.compositionModeDict[str(s)]
            self.img.onImageChanged()

        self.blendingModeCombo.currentIndexChanged.connect(g)

        # shortcut actions
        self.actionDup = QAction('Duplicate layer', None)
        self.actionDup.setShortcut(QKeySequence(Qt.CTRL + Qt.Key_J))
        self.addAction(self.actionDup)
        def dup():
            row = self.selectedIndexes()[0].row()
            #Stack index
            index = len(self.img.layersStack) - row -1
            layer = self.img.layersStack[index]
            if hasattr(layer, 'inputImg'):
                return
            # add new layer to stack and set it to active
            self.img.dupLayer(index=index)
            # update layer view
            self.setLayers(self.img)
            # select the new layer
            # the 1rst call to setActiveLayer has no effect
            # when dup is triggered by the shorcut keys
            # Maybe a Pyside or Qt bug ?
            #self.img.setActiveLayer(index + 1)
            #self.img.setActiveLayer(index + 1)
        self.actionDup.triggered.connect(dup)

    def mousePressEvent(self, e):
        super(QLayerView, self).mousePressEvent(e)
        self.oldMousePos = e.pos()

    def mouseReleaseEvent(self, e):
        # right button click
        if e.pos() == self.oldMousePos:
            if e.button() == Qt.RightButton:
                self.contextMenu(e.pos())
                return
        super(QLayerView, self).mouseReleaseEvent(e)

    def setEnabled(self, value):
        super(QLayerView, self).setEnabled(value)
        if not self.isEnabled():
            self.setStatusTip('Close adjustment form %s to enable Layers' % self.currentWin.windowTitle())
        else:
            self.setStatusTip('')


    def closeAdjustForms(self, delete=False):
        """
        Closes all adjust forms. If delete is True (default False),
        the forms and their dock containers are deleted.
        @param delete:
        @type delete: boolean
        @return: 
        """
        if self.img is None:
            return
        stack = self.img.layersStack
        for layer in  stack:
            if hasattr(layer, "view"):
                if layer.view is not None:
                    dock = layer.view
                    if delete:
                        dock.widget().close()
                        dock.setAttribute(Qt.WA_DeleteOnClose)
                        dock.close()
                        layer.view = None
                    else:
                        dock.close()
        if delete:
            self.currentWin = None

    def clear(self):
        """
        Clears data
        @return: 
        """
        self.closeAdjustForms(delete=True)
        self.img = None
        model = layerModel()
        model.setColumnCount(3)
        self.setModel(None)

    def setLayers(self, mImg):
        """
        Displays the layer stack of mImg
        @param mImg: image
        @type mImg: mImage
        """
        # close open adjustment windows
        self.closeAdjustForms()
        self.img=mImg
        mImg.layerView = self
        model = layerModel()
        model.setColumnCount(3)
        l = len(mImg.layersStack)
        # row edit event handler
        def f(index1, index2):
            #index1 and index2 should be equal
            # only layer name should be editable
            if index1.column() != 1:
                return
            row = index1.row()
            stackIndex = l - row - 1
            mImg.layersStack[stackIndex].name = index1.data()
        model.dataChanged.connect(f)
        for r, lay in enumerate(reversed(mImg.layersStack)):
            items = []
            # col 0 : visibility icon
            if lay.visible :
                item_visible = QStandardItem(QIcon(":/images/resources/eye-icon.png"), "")
            else:
                item_visible = QStandardItem(QIcon(":/images/resources/eye-icon-strike.png"), "")
            items.append(item_visible)
            # col 1 : image icon (for non-adjustment layeronly) and name
            if len(lay.name) <= 12:
                name = lay.name
            else:
                name = lay.name[:10] + '...'
            if hasattr(lay, 'inputImg'):
                item_name = QStandardItem(name)
            else:
                smallImg = lay#.resize(50**3)
                # icon with very small dim causes QPainter error
                # QPixmap.fromImage bug ?
                w,h = smallImg.width(), smallImg.height()
                if w < h / 5 or h < w / 5:
                    item_name = QStandardItem(name)
                else:
                    item_name = QStandardItem(QIcon(QPixmap.fromImage(smallImg)), name)
            # set tool tip to full name
            item_name.setToolTip(lay.name)
            items.append(item_name)
            item_mask = QStandardItem('M')
            items.append(item_mask)
            model.appendRow(items)

        self.setModel(model)

        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        # select active layer
        self.selectRow(len(mImg.layersStack) - 1 - mImg.activeLayerIndex)
        self.update()

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
        @param event:
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
            for srcRow, tgtRow in sorted(rowMapping.items()): # python 3 iteritems->items
                for col in range(0, colCount):
                    self.model().setItem(tgtRow, col, self.model().takeItem(srcRow, col))
            for row in reversed(sorted(rowMapping.keys())): # python 3 iterkeys -> keys
                self.model().removeRow(row)

            rStack = self.img.layersStack[::-1]
            for _ in range(len(rows)):
                rStack.insert(targetRow, None)
            for srcRow, tgtRow in sorted(rowMapping.items()): # python 3 iteritems->items
                rStack[tgtRow] = rStack[srcRow]
            for row in reversed(sorted(rowMapping.keys())): # python 3 iterkeys -> keys
                rStack.pop(row)
            self.img.layersStack = rStack[::-1]
            #event.accept()

    def viewClicked(self, clickedIndex):
        """
        Mouse click event handler.
        @param clickedIndex: 
        @type clickedIndex: QModelIndex
        """
        row = clickedIndex.row()
        layer = self.img.layersStack[-1 - row]
        # toggle layer visibility
        if clickedIndex.column() == 0 :
            # background layer is always visible
            if row == len(self.img.layersStack) - 1:
                return
            #layer = self.img.layersStack[-1-row]
            layer.visible = not(layer.visible)
            # update visibility icon
            if layer.visible:
                self.model().setData(clickedIndex, QIcon(":/images/resources/eye-icon.png") ,Qt.DecorationRole)
            else:
                self.model().setData(clickedIndex, QIcon(":/images/resources/eye-icon-strike.png"), Qt.DecorationRole)
            #if hasattr(layer, 'inputImg'):
                # adjustment layer
                #layer.applyToStack()
            layer.applyToStack()
            # image changed event handler
            self.img.onImageChanged()
        # hide/display adjustment form
        elif clickedIndex.column() == 1 :
            # make selected layer the active layer
            self.img.setActiveLayer(len(self.img.layersStack) - 1 - row)
            opacity = int(self.img.getActiveLayer().opacity * 100)
            self.opacityValue.setText(str('%d ' % opacity))
            self.wdgt.setSliderPosition(opacity)
            compositionMode = self.img.getActiveLayer().compositionMode
            ind = self.blendingModeCombo.findData(compositionMode)
            self.blendingModeCombo.setCurrentIndex(ind)
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
                # make self.currentWin the active window
                self.currentWin.activateWindow()
            # draw the right rectangle
            QtGui1.window.label.repaint()
        # select mask
        elif clickedIndex.column() == 2:
            self.img.layersStack[-1-clickedIndex.row()].maskIsSelected = not self.img.layersStack[-1-clickedIndex.row()].maskIsSelected
            # update
            layer.applyToStack()
            self.img.onImageChanged()

    def contextMenu(self, pos):
        """
        context menu
        @param pos: event coordinates (relative to widget)
        @type pos: QPoint
        """
        index = self.indexAt(pos)
        layer = self.img.layersStack[-1-index.row()]
        lower = self.img.layersStack[layer.getLowerVisibleStackIndex()]
        menu = QMenu()
        actionMerge = QAction('Merge Lower', None)
        # merge only adjustment layer with image layer
        if not hasattr(layer, 'inputImg') or hasattr(lower, 'inputImg'):
            actionMerge.setEnabled(False)
        actionColorMaskEnable = QAction('Color Mask', None)
        actionOpacityMaskEnable = QAction('Opacity Mask', None)
        actionMaskDisable = QAction('Disable Mask', None)
        actionMaskInvert = QAction('Invert Mask', None)
        actionMaskReset = QAction('Clear Mask', None)
        actionMaskCopy = QAction('Copy Mask to Clipboard', None)
        actionMaskPaste = QAction('Paste Mask', None)
        actionMaskDilate = QAction('Dilate Mask', None)
        actionMaskErode = QAction('Erode Mask', None)
        actionMaskPaste.setEnabled(not QApplication.clipboard().image().isNull())
        actionColorMaskEnable.setCheckable(True)
        actionOpacityMaskEnable.setCheckable(True)
        actionColorMaskEnable.setChecked(layer.maskIsSelected and layer.maskIsEnabled)
        actionOpacityMaskEnable.setChecked((not layer.maskIsSelected) and layer.maskIsEnabled)
        # to link actionDup with a shortcut
        # it must be set in __init__
        menu.addAction(self.actionDup)
        # don't dup adjustment layers
        if hasattr(layer, 'inputImg'):
            self.actionDup.setEnabled(False)
        menu.addAction(actionMerge)
        subMenuEnable = menu.addMenu('Enable Mask')
        subMenuEnable.addAction(actionColorMaskEnable)
        subMenuEnable.addAction(actionOpacityMaskEnable)
        menu.addAction(actionMaskDisable)
        menu.addAction(actionMaskInvert)
        menu.addAction(actionMaskReset)
        menu.addAction(actionMaskCopy)
        menu.addAction(actionMaskPaste)
        menu.addAction(actionMaskDilate)
        menu.addAction(actionMaskErode)
        # Event handlers
        def f():
            self.wdgt.show()
        def g(value):
            layer.setOpacity(value)
        def merge():
            layer.merge_with_layer_immediately_below()
        def colorMaskEnable():
            layer.maskIsEnabled = True
            layer.maskIsSelected = True
            layer.applyToStack()
            self.img.onImageChanged()
        def opacityMaskEnable():
            layer.maskIsEnabled = True
            layer.maskIsSelected = False
            layer.applyToStack()
            self.img.onImageChanged()
        def maskDisable():
            layer.maskIsEnabled = False
            layer.maskIsSelected = False
            layer.applyToStack()
            self.img.onImageChanged()
        def maskInvert():
            buf = QImageBuffer(layer.mask)
            buf[:,:,3] = 255 - buf[:,:,3]
            layer.applyToStack()
            self.img.onImageChanged()
        def maskReset():
            layer.resetMask()
            layer.applyToStack()
            self.img.onImageChanged()
        def maskCopy():
            QApplication.clipboard().setImage(layer.mask)
        def maskPaste():
            layer.mask = QApplication.clipboard().image()
            layer.applyToStack()
            self.img.onImageChanged()
        def maskDilate():
            kernel = np.ones((5, 5), np.uint8)
            buf = QImageBuffer(layer.mask)
            # CAUTION erode decreases opacity (min filter), so it extends the masked part of the image
            buf[:, :, 3] = cv2.erode(buf[:,:,3], kernel, iterations=1)
            layer.updatePixmap()
            self.img.onImageChanged()
        def maskErode():
            kernel = np.ones((5, 5), np.uint8)
            buf = QImageBuffer(layer.mask)
            # CAUTION dilate increases opacity (max filter), so it reduces the masked part of the image
            buf[:,:,3] = cv2.dilate(buf[:,:,3], kernel, iterations=1)
            layer.updatePixmap()
            self.img.onImageChanged()
        actionMerge.triggered.connect(merge)
        actionColorMaskEnable.triggered.connect(colorMaskEnable)
        actionOpacityMaskEnable.triggered.connect(opacityMaskEnable)
        actionMaskDisable.triggered.connect(maskDisable)
        actionMaskInvert.triggered.connect(maskInvert)
        actionMaskReset.triggered.connect(maskReset)
        actionMaskCopy.triggered.connect(maskCopy)
        actionMaskPaste.triggered.connect(maskPaste)
        actionMaskDilate.triggered.connect(maskDilate)
        actionMaskErode.triggered.connect(maskErode)
        self.wdgt.valueChanged.connect(g)
        menu.exec_(self.mapToGlobal(pos))


