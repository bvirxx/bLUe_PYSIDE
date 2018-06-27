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
import gc
from PySide2 import QtCore

import cv2
import numpy as np
from PySide2.QtCore import QRectF, QSize, Qt, QModelIndex
from PySide2.QtGui import QImage, QPalette, QKeySequence, QFontMetrics, QTextOption, QPixmap, QIcon, QPainter, QStandardItem, QStandardItemModel
from PySide2.QtWidgets import QAction, QMenu, QSlider, QStyle, QCheckBox, QMessageBox, QApplication, QFileDialog
from PySide2.QtWidgets import QComboBox, QHBoxLayout, QLabel, QTableView, QAbstractItemView, QStyledItemDelegate, QHeaderView, QVBoxLayout
import resources_rc  # hidden import mandatory : DO NOT REMOVE !!!
import QtGui1
from imgconvert import QImageBuffer
from utils import openDlg, dlgWarn


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
        """
        paint event handler
        @param painter:
        @type painter:
        @param option:
        @type option:
        @param index:
        @type index:
        """
        rect = QRectF(option.rect)
        layer = self.parent().img.layersStack[-1 - index.row()]
        # mask
        if index.column() == 2:
            if self.parent().img is not None:
                if layer.maskIsSelected:
                    text = index.data() + ' *'
                else:
                    text = index.data() + '  '
                if layer.group:
                    text = text + ' |'
                if layer.maskIsEnabled:
                    painter.save()
                    painter.setPen(Qt.red)
                    painter.drawText(rect, text, QTextOption(Qt.AlignCenter))
                    painter.restore()
                    return
                painter.drawText(rect, text, QTextOption(Qt.AlignCenter))
        # visibility
        elif index.column() == 0:
            painter.save()
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
        # mouse click event
        self.clicked.connect(self.viewClicked)
        # behavior and style for selection
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        delegate = itemDelegate(parent=self)
        self.setItemDelegate(delegate)
        ic1 = QImage(":/images/resources/eye-icon.png")
        ic2 = QImage(":/images/resources/eye-icon-strike.png")#.scaled(10,20)
        delegate.px1 = QPixmap.fromImage(ic1)
        delegate.px2 = QPixmap.fromImage(ic2)
        ic1.invertPixels()
        ic2.invertPixels()
        delegate.inv_px1 = QPixmap.fromImage(ic1)
        delegate.inv_px2 = QPixmap.fromImage(ic2)
        self.setIconSize(QSize(20,15))
        self.verticalHeader().setMinimumSectionSize(-1)
        self.verticalHeader().setDefaultSectionSize(self.verticalHeader().minimumSectionSize())
        self.horizontalHeader().setMinimumSectionSize(40)
        self.horizontalHeader().setDefaultSectionSize(40)
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
        self.previewOptionBox = QCheckBox('Preview')
        self.previewOptionBox.setMaximumSize(100, 30)
        # View/Preview changed event handler
        def m(state): # state : Qt.Checked Qt.UnChecked
            if self.img is None:
                return
            self.img.useThumb = (state == Qt.Checked)
            QtGui1.window.updateStatus()
            self.img.cacheInvalidate()
            for l in self.img.layersStack:
                l.autoclone = True  # auto update cloning layers
                l.knitted = False
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor) #TODO 18/04/18 waitcursor is called by applytostack?
                QtGui1.app.processEvents()
                # update whole stack
                self.img.layersStack[0].applyToStack()
            finally:
                for l in self.img.layersStack:
                    l.autoclone = False  # reset flags
                    l.knitted = False
                QApplication.restoreOverrideCursor()
                QApplication.processEvents()
            QtGui1.window.label.repaint()
        self.previewOptionBox.stateChanged.connect(m)
        self.previewOptionBox.setChecked(True) # m is not triggered

        # opacity slider
        self.opacitySlider = QSlider(Qt.Horizontal)
        self.opacitySlider.setTickPosition(QSlider.TicksBelow)
        self.opacitySlider.setRange(0, 100)
        self.opacitySlider.setSingleStep(1)
        self.opacitySlider.setSliderPosition(100)
        #self.opacitySlider.setTracking(False)
        opacityLabel = QLabel()
        opacityLabel.setMaximumSize(100,30)
        #self.opacityLabel.setStyleSheet("QLabel {background-color: white;}")
        opacityLabel.setText("Layer opacity")
        hl0 = QHBoxLayout()
        hl0.addWidget(opacityLabel)
        # l.addWidget(opacityLabel)
        hl0.addStretch(1)
        hl0.addWidget(self.previewOptionBox)
        l.addLayout(hl0)

        hl =  QHBoxLayout()
        self.opacityValue = QLabel()
        font = self.opacityValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("100 ")
        h = metrics.height()
        self.opacityValue.setMinimumSize(w, h)
        self.opacityValue.setMaximumSize(w, h)

        self.opacityValue.setText('100 ')
        self.opacityValue.setStyleSheet("QLabel {background-color: white}")
        hl.addWidget(self.opacityValue)
        hl.addWidget(self.opacitySlider)
        l.addLayout(hl)
        l.setContentsMargins(20,0,20,0) # left, top, right, bottom
        # the layout is set in blue.py, after the initialization of the main form.
        self.propertyLayout = l
        # opacity value changed event handler
        def f1():
            self.opacityValue.setText(str('%d ' % self.opacitySlider.value()))
        # slider released event handler
        def f2():
            self.img.getActiveLayer().setOpacity(self.opacitySlider.value())
            self.img.getActiveLayer().applyToStack()
            self.img.onImageChanged()
        self.opacitySlider.valueChanged.connect(f1)
        self.opacitySlider.sliderReleased.connect(f2)
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
        # combo box item chosen event handler
        def g(ind):
            s = self.blendingModeCombo.currentText()
            self.img.getActiveLayer().compositionMode = self.compositionModeDict[str(s)]
            self.img.getActiveLayer().applyToStack()
            self.img.onImageChanged()
        #self.blendingModeCombo.currentIndexChanged.connect(g1)
        self.blendingModeCombo.activated.connect(g)
        # shortcut actions
        self.actionDup = QAction('Duplicate layer', None)
        self.actionDup.setShortcut(QKeySequence(Qt.CTRL + Qt.Key_J))
        self.addAction(self.actionDup)
        def dup():
            row = self.selectedIndexes()[0].row()
            #Stack index
            index = len(self.img.layersStack) - row -1
            layer = self.img.layersStack[index]
            if layer.isAdjustLayer():
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
        self.setWhatsThis(
"""
To toggle the layer visibility click on the Eye icon.
To add a mask use the context menu to enable it and paint pixels with the Mask/Unmask tools found in the left pane.
For color mask mode : 
    grey pixels are masked,
    reddish pixels are unmasked.
Note that upper visible layers slow down mask edition.
You can drag and drop layers to change their order.
"""
        )

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
        """
        if self.img is None:
            return
        stack = self.img.layersStack
        for layer in  stack:
            if hasattr(layer, "view"):
                if layer.view is not None:
                    dock = layer.view
                    if delete:
                        # remove back link
                        dock.widget().layer = None
                        QtGui1.window.removeDockWidget(dock)
                        dock.widget().setAttribute(Qt.WA_DeleteOnClose)
                        dock.widget().deleteLater()
                        dock.widget().close()
                        dock.setAttribute(Qt.WA_DeleteOnClose)
                        dock.deleteLater()
                        dock.close()
                        layer.view = None
                    else:
                        dock.close()
        if delete:
            self.currentWin = None
            gc.collect()

    def clear(self, delete=True):
        """
        Resets LayerView and clears back
        links to image
        @return: 
        """
        self.closeAdjustForms(delete=delete) #TODO modified 8/10/17 for merge_with_layer_immediatly_below
        self.img = None
        self.currentWin = None
        model = layerModel()
        model.setColumnCount(3)
        self.setModel(None)

    def setLayers(self, mImg, delete=False):
        """
        Displays the layer stack of mImg
        @param mImg: image
        @type mImg: mImage
        """
        # close open adjustment windows
        #self.closeAdjustForms()
        self.clear(delete=delete)  # TODO 01/12/17 switched from closeadjust to clear
        # double link
        self.img = mImg
        mImg.layerView = self
        model = layerModel()
        model.setColumnCount(3)
        l = len(mImg.layersStack)
        # dataChanged event handler : enables edition of layer name
        def f(index1, index2):
            #index1 and index2 should be equal
            # only layer name should be editable
            # dropEvent emit dataChanged when setting item values. f must
            # return immediately from these calls, as they are possibly made with unconsistent data :
            # dragged rows are already removed from layersStack
            # and not yet removed from model.
            if l != self.model().rowCount():
                return
            # only name is editable
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
            if len(lay.name) <= 30:
                name = lay.name
            else:
                name = lay.name[:28] + '...'
            if hasattr(lay, 'inputImg'):
                item_name = QStandardItem(name)
            else:
                # icon with very small dim causes QPainter error
                # QPixmap.fromImage bug ?
                smallImg = lay.resize(50 * 50)
                w,h = smallImg.width(), smallImg.height()
                if w < h / 5 or h < w / 5:
                    item_name = QStandardItem(name)
                else:
                    item_name = QStandardItem(QIcon(QPixmap.fromImage(smallImg)), name)
            # set tool tip to full name
            item_name.setToolTip(lay.name)
            items.append(item_name)
            item_mask = QStandardItem('m')
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
        self.updateForm()
        for item in self.img.layersStack:
            if hasattr(item, 'sourceIndex'):
                combo = item.view.widget().sourceCombo
                currentText = combo.currentText()
                combo.clear()
                for i, x in enumerate(self.img.layersStack):
                    item.view.widget().sourceCombo.addItem(x.name, i)
                combo.setCurrentIndex(combo.findText(currentText))

    def updateForm(self):
        activeLayer = self.img.getActiveLayer()
        if hasattr(activeLayer, 'view'):
            self.currentWin = activeLayer.view
        if self.currentWin is not None:
            self.currentWin.show()
            self.currentWin.activateWindow()

    def updateRow(self, row):
        minInd, maxInd = self.model().index(row, 0), self.model().index(row, 3)
        self.model().dataChanged.emit(minInd, maxInd)

    def dropEvent(self, event):
        """
        drop event handler : moving layer
        @param event:
        @type event: Qevent
        """
        if event.source() is not self:
            return
        # get selected rows and layers
        rows = set([mi.row() for mi in self.selectedIndexes()])
        rStack = self.img.layersStack[::-1]
        layers = [rStack[i] for i in rows]
        linked = any(l.group for l in layers)
        if linked and len(rows)> 1:
            return
        # get target row and layer
        targetRow = self.indexAt(event.pos()).row()
        targetLayer = rStack[targetRow]
        if linked:
            if layers[0].group is not targetLayer.group:
                return
        if bool(targetLayer.group) != linked:
            return
        # remove target from selection
        if targetRow in rows:
            rows.discard(targetRow)
        rows = sorted(rows)
        if not rows:
            return
        # if target is below last row insert at the last position
        if targetRow == -1:
            targetRow = self.model().rowCount()
        # mapping of src (row) indices to target indices
        rowMapping = dict()
        for idx, row in enumerate(rows):
            if row < targetRow:
                rowMapping[row] = targetRow + idx
            else:
                rowMapping[row + len(rows)] = targetRow + idx
        # update layerStack using rowMapping
        # insert None items
        for _ in range(len(rows)):
            rStack.insert(targetRow, None)
        # copy moved items to their final place
        for srcRow, tgtRow in sorted(rowMapping.items()):  # python 3 iteritems->items
            rStack[tgtRow] = rStack[srcRow]
        # remove moved items from their initial place
        for row in reversed(sorted(rowMapping.keys())):  # python 3 iterkeys -> keys
            rStack.pop(row)
        self.img.layersStack = rStack[::-1]
        # update model
        # insert empty rows
        for _ in range(len(rows)):
            result = self.model().insertRow(targetRow, QModelIndex())
        # copy moved rows to their final place
        colCount = self.model().columnCount()
        for srcRow, tgtRow in sorted(rowMapping.items()): # python 3 iteritems->items
            for col in range(0, colCount):
                # CAUTION : setItem calls the data changed event handler (cf. setLayers above)
                self.model().setItem(tgtRow, col, self.model().takeItem(srcRow, col))
        # remove moved rows from their initial place and keep track of moved items
        movedDict = rowMapping.copy()
        for row in reversed(sorted(rowMapping.keys())): # python 3 iterkeys -> keys
            self.model().removeRow(row)
            for s, t in rowMapping.items():
                if t > row:
                    movedDict[s]-=1
        ######################################### sanity check
        for r in range(self.model().rowCount()):
            id = self.model().index(r, 1)
            if id.data() != rStack[r].name:
                raise ValueError('Drop Error')
        ########################################
        # reselect moved rows
        sel = sorted(movedDict.values())
        selectionModel = QtCore.QItemSelectionModel(self.model())
        self.setSelectionModel(selectionModel)
        index1 = self.model().index(sel[0], 1)
        index2 = self.model().index(sel[-1], 1)
        itemSelection = QtCore.QItemSelection(index1, index2)
        self.selectionModel().select(itemSelection,  QtCore.QItemSelectionModel.Rows | QtCore.QItemSelectionModel.Select)
        # multiple selection : display no window
        if len(sel) > 1 :
            self.currentWin.hide()
            self.currentWin = None
        elif len(sel) == 1:
            self.img.setActiveLayer(len(self.img.layersStack) - sel[0] -1)
        # update stack
        self.img.layersStack[0].applyToStack()
        self.img.onImageChanged()

    def select(self, row, col):
        """
        select item in view
        @param row:
        @type row:
        @param col:
        @type col:
        @return:
        @rtype:
        """
        model = self.model()
        self.viewClicked(model.index(row, col))

    def viewClicked(self, clickedIndex):
        """
        Mouse clicked event handler.
        @param clickedIndex: 
        @type clickedIndex: QModelIndex
        """
        row = clickedIndex.row()
        rows = set([mi.row() for mi in self.selectedIndexes()])
        #multiple selection : go to top of selection
        m = min(rows)
        if row != m :
            clickedIndex = self.model().index(m, clickedIndex.column())
        layer = self.img.layersStack[-1 - row]
        self.actionDup.setEnabled(not layer.isAdjustLayer())
        # toggle layer visibility
        if clickedIndex.column() == 0 :
            # background layer is always visible
            if row == len(self.img.layersStack) - 1:
                return
            #layer.visible = not(layer.visible)
            layer.setVisible(not(layer.visible))
            # update visibility icon
            if layer.visible:
                self.model().setData(clickedIndex, QIcon(":/images/resources/eye-icon.png") ,Qt.DecorationRole)
            else:
                self.model().setData(clickedIndex, QIcon(":/images/resources/eye-icon-strike.png"), Qt.DecorationRole)
            if self.currentWin is not None:
                self.currentWin.setVisible(layer.visible)
                if not layer.visible:
                    self.currentWin = None
            if layer.tool is not None:
                layer.tool.setVisible(layer.visible)
            # update stack
            if layer.visible:
                layer.applyToStack()
            else:
                i = layer.getUpperVisibleStackIndex()
                if i >=0:
                    layer.parentImage.layersStack[i].applyToStack()
                else:
                    # top layer : update only the presentation layer
                    layer.parentImage.prLayer.execute(l=None, pool=None)
            self.img.onImageChanged()
        # hide/display adjustment form
        elif clickedIndex.column() == 1 :
            pass
        elif clickedIndex.column() == 2:
            pass
            """
            cl = self.img.layersStack[-1-clickedIndex.row()]
            cl.maskIsSelected = not cl.maskIsSelected
            layer.applyToStack()
            self.img.onImageChanged()
            """
        # update displayed window and active layer
        activeStackIndex = len(self.img.layersStack) - 1 - row
        activeLayer = self.img.setActiveLayer(activeStackIndex)
        if self.currentWin is not None:
            self.currentWin.hide()
            self.currentWin = None
        if hasattr(self.img.layersStack[activeStackIndex], "view"):
            self.currentWin = self.img.layersStack[activeStackIndex].view
        if self.currentWin is not None and activeLayer.visible:
            self.currentWin.show()
            # make self.currentWin the active window
            self.currentWin.activateWindow()
        # update opacity and composition mode for current layer
        opacity = int(layer.opacity * 100)
        self.opacityValue.setText(str('%d ' % opacity))
        self.opacitySlider.setSliderPosition(opacity)
        compositionMode = layer.compositionMode
        ind = self.blendingModeCombo.findData(compositionMode)
        self.blendingModeCombo.setCurrentIndex(ind)
        # draw the right rectangle
        QtGui1.window.label.repaint()

    def initContextMenu(self):
        """
        return context menu
        @return:
        @rtype: QMenu
        """
        menu = QMenu()
        menu.actionReset = QAction('Reset To Default', None)
        menu.actionLoadImage = QAction('Load New Image', None)
        menu.actionGroupSelection = QAction('Group Selection', None)
        menu.actionAdd2Group = QAction('Add to Group', None)
        # Active layer is not in a group or right clicked layer is in a group

        menu.actionUnGroup = QAction('Ungroup', None)

        # multiple selection
        menu.actionMerge = QAction('Merge Lower', None)
        # merge only adjustment layer with image layer

        # don't dup adjustment layers

        menu.actionUnselect = QAction('Unselect All', None)

        menu.actionRepositionLayer = QAction('Reposition Layer(s)', None)
        menu.actionColorMaskEnable = QAction('Color Mask', None)
        menu.actionOpacityMaskEnable = QAction('Opacity Mask', None)
        menu.actionMaskDisable = QAction('Disable Mask', None)
        menu.actionMaskInvert = QAction('Invert Mask', None)
        menu.actionMaskReset = QAction('Clear Mask', None)
        menu.actionMaskCopy = QAction('Copy Mask to Clipboard', None)
        menu.actionImageCopy = QAction('Copy Image to Clipboard', None)
        menu.actionMaskPaste = QAction('Paste Mask', None)
        menu.actionImagePaste = QAction('Paste Image', None)
        menu.actionMaskDilate = QAction('Dilate Mask', None)
        menu.actionMaskErode = QAction('Erode Mask', None)
        menu.actionColorMaskEnable.setCheckable(True)
        menu.actionOpacityMaskEnable.setCheckable(True)
        ####################
        # Build menu
        ###################
        # group/ungroup
        menu.addAction(menu.actionAdd2Group)
        menu.addAction(menu.actionGroupSelection)
        menu.addAction(menu.actionUnGroup)
        menu.addSeparator()
        menu.addAction(menu.actionUnselect)
        menu.addSeparator()
        menu.addAction(menu.actionRepositionLayer)
        menu.addSeparator()
        # layer
        menu.addAction(menu.actionImageCopy)
        menu.addAction(menu.actionImagePaste)
        menu.addSeparator()
        # mask
        menu.subMenuEnable = menu.addMenu('Enable Mask As...')
        menu.subMenuEnable.addAction(menu.actionColorMaskEnable)
        menu.subMenuEnable.addAction(menu.actionOpacityMaskEnable)
        menu.addAction(menu.actionMaskDisable)
        menu.addAction(menu.actionMaskInvert)
        menu.addAction(menu.actionMaskReset)
        menu.addAction(menu.actionMaskCopy)
        menu.addAction(menu.actionMaskPaste)
        menu.addAction(menu.actionMaskDilate)
        menu.addAction(menu.actionMaskErode)
        menu.addSeparator()
        # miscellaneous
        menu.addAction(menu.actionLoadImage)
        # to link actionDup with a shortcut,
        # it must be set in __init__
        menu.addAction(self.actionDup)
        menu.addAction(menu.actionMerge)
        menu.addAction(menu.actionReset)
        return menu

    def contextMenuEvent(self, event):
        """
        context menu
        @param event
        @type event: QContextMenuEvent
        """
        selection = self.selectedIndexes()
        if not selection:
            return
        # get fresh context menu
        self.cMenu = self.initContextMenu()
        # get current selection
        rows = set([mi.row() for mi in selection])
        rStack = self.img.layersStack[::-1]
        layers = [rStack[r] for r in rows]
        if layers:
            group = layers[0].group
        for l in layers:
            # different groups
            if l.group and group:
                if l.group is not group:
                    dlgWarn("Select a single group")
                    return
        # get current position
        index = self.indexAt(event.pos())
        layerStackIndex = len(self.img.layersStack) - 1 - index.row()
        layer = self.img.layersStack[layerStackIndex]
        lowerVisible = self.img.layersStack[layer.getLowerVisibleStackIndex()]
        lower = self.img.layersStack[layerStackIndex - 1]  # case index == 0 doesn't matter
        # toggle actions
        self.cMenu.actionGroupSelection.setEnabled(not(len(rows) < 2 or any(l.group for l in layers)))
        self.cMenu.actionAdd2Group.setEnabled(not(group or layer.group))
        self.cMenu.actionUnGroup.setEnabled(bool(layer.group))
        self.cMenu.actionMerge.setEnabled(not (hasattr(layer, 'inputImg') or hasattr(lowerVisible, 'inputImg')))
        self.actionDup.setEnabled(not layer.isAdjustLayer())
        self.cMenu.actionColorMaskEnable.setChecked(layer.maskIsSelected and layer.maskIsEnabled)
        self.cMenu.actionOpacityMaskEnable.setChecked((not layer.maskIsSelected) and layer.maskIsEnabled)
        self.cMenu.actionUnselect.setEnabled(layer.rect is None)
        self.cMenu.subMenuEnable.setEnabled(len(rows)==1)
        self.cMenu.actionMaskPaste.setEnabled(not QApplication.clipboard().image().isNull())
        self.cMenu.actionImagePaste.setEnabled(not QApplication.clipboard().image().isNull())
        # Event handlers
        def f():
            self.opacitySlider.show()
        def unselectAll():
            layer.rect = None
        def RepositionLayer():
            layer.xOffset, layer.yOffset = 0, 0
            layer.Zoom_coeff = 1.0
            layer.AltZoom_coeff = 1.0
            layer.xAltOffset, layer.yAltOffset = 0, 0
            layer.updatePixmap()
            self.img.onImageChanged()
        def loadImage():
            return # TODO 26/06/18 action to remove from menu? replaced by new image layer
            window = QtGui1.window
            filename = openDlg(window)
            img = QImage(filename)
            layer.thumb = None
            layer.setImage(img)
        def add2Group():
            layer.group = group
            layer.mask = group[0].mask
            layer.maskIsEnabled = True
            layer.maskIsSelected = True
        def groupSelection():
            layers = [rStack[i] for i in sorted(rows)]
            if any(l.group for l in layers):
                dlgWarn("Some layers are already grouped. Ungroup first")
                return
            mask = layers[0].mask
            for l in layers:
                l.group = layers
                l.mask = mask
                l.maskIsEnabled = True
                l.maskIsSelected = False
        def unGroup():
            group = layer.group.copy()
            for l in group:
                l.unlinkMask()
        def merge():
            layer.merge_with_layer_immediately_below()
        def testUpperVisibility():
            pos = self.img.getStackIndex(layer)
            upperVisible = False
            for i in range(len(self.img.layersStack) - pos - 1):
                if self.img.layersStack[pos + 1 + i].visible:
                    upperVisible = True
                    break
            if upperVisible:
                dlgWarn("Upper visible layers slow down mask edition")
                return True
            return False
        def colorMaskEnable():
            testUpperVisibility()
            layer.maskIsEnabled = True
            layer.maskIsSelected = True
            layer.applyToStack()
            self.img.onImageChanged()
        def opacityMaskEnable():
            testUpperVisibility()
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
            layer.invertMask()
            # update mask stack
            layer.applyToStack()
            #for l in self.img.layersStack:
                #l.updatePixmap(maskOnly=True)
            self.img.onImageChanged()
        def maskReset():
            layer.resetMask()
            # update mask stack
            for l in self.img.layersStack:
                l.updatePixmap(maskOnly=True)
            self.img.onImageChanged()
        def maskCopy():
            QApplication.clipboard().setImage(layer.mask)
        def imageCopy():
            QApplication.clipboard().setImage(layer.getCurrentMaskedImage())
        def maskPaste():
            """
            Pastes clipboard to mask and updates the stack. The clipboard image
            is scaled if its size does not match the size of the mask
            """
            cb = QApplication.clipboard()
            if not cb.image().isNull():
                img = cb.image()
                if img.size() == layer.mask.size():
                    layer.mask = img
                else:
                    layer.mask = img.scaled(layer.mask.size())
            layer.applyToStack()
            self.img.onImageChanged()
        def imagePaste():
            """
            Pastes clipboard to mask and updates the stack. The clipboard image
            is scaled if its size does not match the size of the mask
            """
            cb = QApplication.clipboard()
            if not cb.image().isNull():
                srcImg = cb.image()
                if srcImg.size() == layer.size():
                    layer.setImage(srcImg)
                else:
                    layer.setImage(srcImg.scaled(layer.size()))
            layer.applyToStack()
            self.img.onImageChanged()
        def maskDilate():
            kernel = np.ones((5, 5), np.uint8)
            buf = QImageBuffer(layer.mask)
            # CAUTION erode decreases values (min filter), so it extends the masked part of the image
            buf[:, :, 2] = cv2.erode(buf[:,:,2], kernel, iterations=1)
            for l in self.img.layersStack:
                l.updatePixmap(maskOnly=True)
            self.img.onImageChanged()
        def maskErode():
            kernel = np.ones((5, 5), np.uint8)
            buf = QImageBuffer(layer.mask)
            # CAUTION dilate increases values (max filter), so it reduces the masked part of the image
            buf[:,:,2] = cv2.dilate(buf[:,:,2], kernel, iterations=1)
            for l in self.img.layersStack:
                l.updatePixmap(maskOnly=True)
            self.img.onImageChanged()
        def layerReset():
            view = layer.view.widget()
            if hasattr(view, 'setDefaults'):
                view.setDefaults()

        self.cMenu.actionRepositionLayer.triggered.connect(RepositionLayer)
        self.cMenu.actionUnselect.triggered.connect(unselectAll)
        self.cMenu.actionLoadImage.triggered.connect(loadImage)
        self.cMenu.actionAdd2Group.triggered.connect(add2Group)
        self.cMenu.actionGroupSelection.triggered.connect(groupSelection)
        self.cMenu.actionUnGroup.triggered.connect(unGroup)
        self.cMenu.actionMerge.triggered.connect(merge)
        self.cMenu.actionColorMaskEnable.triggered.connect(colorMaskEnable)
        self.cMenu.actionOpacityMaskEnable.triggered.connect(opacityMaskEnable)
        self.cMenu.actionMaskDisable.triggered.connect(maskDisable)
        self.cMenu.actionMaskInvert.triggered.connect(maskInvert)
        self.cMenu.actionMaskReset.triggered.connect(maskReset)
        self.cMenu.actionMaskCopy.triggered.connect(maskCopy)
        self.cMenu.actionMaskPaste.triggered.connect(maskPaste)
        self.cMenu.actionImageCopy.triggered.connect(imageCopy)
        self.cMenu.actionImagePaste.triggered.connect(imagePaste)
        self.cMenu.actionMaskDilate.triggered.connect(maskDilate)
        self.cMenu.actionMaskErode.triggered.connect(maskErode)
        self.cMenu.actionReset.triggered.connect(layerReset)
        self.cMenu.exec_(event.globalPos())
        # update table
        for row in rows:
            self.updateRow(row)

