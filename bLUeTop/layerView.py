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
from collections import OrderedDict

from PySide2 import QtCore
from PySide2.QtCore import QRectF, QSize, Qt, QModelIndex, QPoint
from PySide2.QtGui import QImage, QPalette, QKeySequence, QFontMetrics, QTextOption, QPixmap, QIcon, QPainter, QStandardItem, QStandardItemModel
from PySide2.QtWidgets import QAction, QMenu, QSlider, QStyle, QCheckBox, QApplication
from PySide2.QtWidgets import QComboBox, QHBoxLayout, QLabel, QTableView, QAbstractItemView, QStyledItemDelegate, QHeaderView, QVBoxLayout

from bLUeTop.QtGui1 import window
from bLUeGui.bLUeImage import QImageBuffer
from bLUeGui.dialog import dlgWarn
from bLUeGui.memory import weakProxy
from bLUeTop.settings import TABBING
from bLUeTop.utils import QbLUeSlider
from bLUeTop.versatileImg import vImage


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
        # mask and merged layer flag (for exposure fusion and HDR merge)
        if index.column() == 2:
            if self.parent().img is not None:
                if layer.maskIsSelected:
                    text = index.data() + ' *'
                else:
                    text = index.data() + '  '
                if layer.mergingFlag:
                    text = text + ' +'
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
            bgColor = option.palette.color(QPalette.Window)
            bgColor = bgColor.red(), bgColor.green(), bgColor.blue()
            dark = (max(bgColor) <= 128)
            if option.state & QStyle.State_Selected:
                c = option.palette.color(QPalette.Highlight)
                painter.fillRect(rect, c)
            if self.parent().img.layersStack[-1 - index.row()].visible:
                px = self.inv_px1 if dark or (option.state & QStyle.State_Selected) else self.px1
            else:
                px = self.inv_px2 if dark or (option.state & QStyle.State_Selected) else self.px2
            painter.drawPixmap(rect, px, QRectF(0, 0, px.width(), px.height()))
            painter.restore()
        else:
            # call default
            QStyledItemDelegate.paint(self, painter, option, index)


class QLayerView(QTableView):
    """
    Display the stack of image layers.
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.img = None
        # graphic form to show : it
        # should correspond to the currently selected layer
        self.currentWin = None
        # mouse click event
        self.clicked.connect(self.viewClicked)

        # set behavior and styles
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
        self.setIconSize(QSize(20, 15))
        self.verticalHeader().setMinimumSectionSize(-1)
        self.verticalHeader().setDefaultSectionSize(self.verticalHeader().minimumSectionSize())
        self.horizontalHeader().setMinimumSectionSize(40)
        self.horizontalHeader().setDefaultSectionSize(40)

        # drag and drop
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDragDropOverwriteMode(False)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)

        ################################
        # layer property GUI :
        # preview, blending mode, opacity, mask color
        ################################
        # Preview option
        # We should use a QListWidget or a custom optionsWidget
        # (cf. utils.py) :  adding it to QVBoxLayout with mode
        # Qt.AlignBottom does not work.
        self.previewOptionBox = QCheckBox('Preview')
        self.previewOptionBox.setMaximumSize(100, 30)

        # View/Preview changed event handler
        def m(state):  # state : Qt.Checked Qt.UnChecked
            if self.img is None:
                return
            self.img.useThumb = (state == Qt.Checked)
            window.updateStatus()
            self.img.cacheInvalidate()
            # for layer in self.img.layersStack:
                # layer.autoclone = True  # auto update cloning layers
                # layer.knitted = False
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)  # TODO 18/04/18 waitcursor already called by applytostack
                QApplication.processEvents()
                # update the whole stack
                self.img.layersStack[0].applyToStack()
                self.img.onImageChanged()
            finally:
                # for layer in self.img.layersStack:
                    # layer.autoclone = False  # reset flags
                    # layer.knitted = False
                QApplication.restoreOverrideCursor()
                QApplication.processEvents()
        self.previewOptionBox.stateChanged.connect(m)
        self.previewOptionBox.setChecked(True)  # m is not triggered

        # title
        titleLabel = QLabel('Layer')
        titleLabel.setMaximumSize(100, 30)

        # opacity slider
        self.opacitySlider = QbLUeSlider(Qt.Horizontal)
        self.opacitySlider.setStyleSheet(QbLUeSlider.bLueSliderDefaultBWStylesheet)
        self.opacitySlider.setTickPosition(QSlider.TicksBelow)
        self.opacitySlider.setRange(0, 100)
        self.opacitySlider.setSingleStep(1)
        self.opacitySlider.setSliderPosition(100)

        self.opacityValue = QLabel()
        font = self.opacityValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("100 ")
        h = metrics.height()
        self.opacityValue.setMinimumSize(w, h)
        self.opacityValue.setMaximumSize(w, h)
        self.opacityValue.setText('100 ')

        # opacity value changed event handler
        def f1():
            self.opacityValue.setText(str('%d ' % self.opacitySlider.value()))

        # opacity slider released event handler
        def f2():
            try:
                layer = self.img.getActiveLayer()
                layer.setOpacity(self.opacitySlider.value())
                layer.applyToStack()
                self.img.onImageChanged()
            except AttributeError:
                return

        self.opacitySlider.valueChanged.connect(f1)
        self.opacitySlider.sliderReleased.connect(f2)

        # mask color slider
        self.maskLabel = QLabel('Mask Color')
        maskSlider = QbLUeSlider(Qt.Horizontal)
        maskSlider.setStyleSheet(QbLUeSlider.bLueSliderDefaultBWStylesheet)
        maskSlider.setTickPosition(QSlider.TicksBelow)
        maskSlider.setRange(0, 100)
        maskSlider.setSingleStep(1)
        maskSlider.setSliderPosition(100)
        self.maskSlider = maskSlider

        self.maskValue = QLabel()
        font = self.maskValue.font()
        metrics = QFontMetrics(font)
        w = metrics.width("100 ")
        h = metrics.height()
        self.maskValue.setMinimumSize(w, h)
        self.maskValue.setMaximumSize(w, h)
        self.maskValue.setText('100 ')
        # masks are disbled by default
        self.maskLabel.setEnabled(False)
        self.maskSlider.setEnabled(False)
        self.maskValue.setEnabled(False)

        # mask value changed event handler
        def g1():
            self.maskValue.setText(str('%d ' % self.maskSlider.value()))

        # mask slider released event handler
        def g2():
            try:
                layer = self.img.getActiveLayer()
                layer.setColorMaskOpacity(self.maskSlider.value() * 255.0 / 100.0)
                layer.applyToStack()
                self.img.onImageChanged()
            except AttributeError:
                return

        self.maskSlider.valueChanged.connect(g1)
        self.maskSlider.sliderReleased.connect(g2)

        # blending mode combo box
        compLabel = QLabel()
        compLabel.setText("Blend")

        self.compositionModeDict = OrderedDict([('Normal', QPainter.CompositionMode_SourceOver),
                                                ('Plus', QPainter.CompositionMode_Plus),
                                                ('Multiply', QPainter.CompositionMode_Multiply),
                                                ('Screen', QPainter.CompositionMode_Screen),
                                                ('Overlay', QPainter.CompositionMode_Overlay),
                                                ('Darken', QPainter.CompositionMode_Darken),
                                                ('Lighten', QPainter.CompositionMode_Lighten),
                                                ('Color Dodge', QPainter.CompositionMode_ColorDodge),
                                                ('Color Burn', QPainter.CompositionMode_ColorBurn),
                                                ('Hard Light', QPainter.CompositionMode_HardLight),
                                                ('Soft Light', QPainter.CompositionMode_SoftLight),
                                                ('Difference', QPainter.CompositionMode_Difference),
                                                ('Exclusion', QPainter.CompositionMode_Exclusion)
                                                ])
        self.blendingModeCombo = QComboBox()
        for key in self.compositionModeDict:
            self.blendingModeCombo.addItem(key, self.compositionModeDict[key])

        # combo box item chosen event handler
        def g(ind):
            s = self.blendingModeCombo.currentText()
            try:
                layer = self.img.getActiveLayer()
                layer.compositionMode = self.compositionModeDict[str(s)]
                layer.applyToStack()
                self.img.onImageChanged()
            except AttributeError:
                return

        self.blendingModeCombo.currentIndexChanged.connect(g)

        # layout
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignTop)
        hl0 = QHBoxLayout()
        hl0.addWidget(titleLabel)
        hl0.addStretch(1)
        hl0.addWidget(self.previewOptionBox)
        l.addLayout(hl0)
        hl = QHBoxLayout()
        hl.addWidget(QLabel('Opacity'))
        hl.addWidget(self.opacityValue)
        hl.addWidget(self.opacitySlider)
        l.addLayout(hl)
        hl1 = QHBoxLayout()
        hl1.addWidget(self.maskLabel)
        hl1.addWidget(self.maskValue)
        hl1.addWidget(self.maskSlider)
        l.addLayout(hl1)
        l.setContentsMargins(0, 0, 10, 0)  # left, top, right, bottom
        hl2 = QHBoxLayout()
        hl2.addWidget(compLabel)
        hl2.addWidget(self.blendingModeCombo)
        l.addLayout(hl2)
        for layout in [hl, hl1, hl2]:
            layout.setContentsMargins(5, 0, 0, 0)
        # this layout must be added to the propertyWidget object loaded from blue.ui :
        # we postpone it to in blue.py, after loading the main form.
        self.propertyLayout = l
        # shortcut actions
        self.actionDup = QAction('Duplicate layer', None)
        self.actionDup.setShortcut(QKeySequence(Qt.CTRL + Qt.Key_J))
        self.addAction(self.actionDup)

        def dup():
            row = self.selectedIndexes()[0].row()
            # Stack index
            index = len(self.img.layersStack) - row - 1
            layer = self.img.layersStack[index]
            if layer.isAdjustLayer():
                return
            # add new layer to stack and set it to active
            self.img.dupLayer(index=index)
            # update layer view
            self.setLayers(self.img)
        self.actionDup.triggered.connect(dup)
        self.setWhatsThis(
"""<b>Layer Stack</b>
To <b>toggle the layer visibility</b> click on the Eye icon.<br>
To <b>add a mask</b> use the context menu to enable it and paint pixels with the Mask/Unmask tools in the left pane.<br>
For <b>color mask<b/b>: <br>
    &nbsp; green pixels are masked,<br>
    &nbsp; red pixels are unmasked.<br>
Note that upper visible layers slow down mask edition.<br>
"""
                        )  # end of setWhatsThis

    def closeAdjustForms(self, delete=False):
        """
        Close all layer forms. If delete is True (default False),
        the forms and their dock containers are deleted.
        @param delete:
        @type delete: boolean
        """
        if self.img is None:
            return
        stack = self.img.layersStack
        for layer in stack:
            if hasattr(layer, "view"):
                if layer.view is not None:
                    dock = layer.view
                    if delete:
                        form = dock.widget()
                        # remove back link
                        form.layer = None
                        # QtGui1.window.removeDockWidget(dock)
                        form.setAttribute(Qt.WA_DeleteOnClose)
                        form.close()
                        dock.setAttribute(Qt.WA_DeleteOnClose)
                        dock.close()
                        layer.view = None
                    elif not TABBING:  # tabbed forms should not be closed
                        dock.close()
        if delete:
            self.currentWin = None
            gc.collect()

    def clear(self, delete=True):
        """
        Reset LayerView and clear the back
        links to image.
        """
        self.closeAdjustForms(delete=delete)
        self.img = None
        self.currentWin = None
        model = layerModel()
        model.setColumnCount(3)
        self.setModel(None)

    def setLayers(self, mImg, delete=False):
        """
        Displays the layer stack of a mImage instance.
        @param mImg: image
        @type mImg: mImage
        @param delete:
        @type delete:
        """
        # close open adjustment windows
        # self.closeAdjustForms()
        self.clear(delete=delete)
        mImg.layerView = self
        # back link to image
        self.img = weakProxy(mImg)
        model = layerModel()
        model.setColumnCount(3)
        l = len(mImg.layersStack)

        # dataChanged event handler : enables edition of layer name
        def f(index1, index2):
            # index1 and index2 should be equal
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
            try:
                lay.maskSettingsChanged.sig.disconnect()
            except RuntimeError:
                pass
            lay.maskSettingsChanged.sig.connect(self.updateRows)
            items = []
            # col 0 : visibility icon
            if lay.visible:
                item_visible = QStandardItem(QIcon(":/images/resources/eye-icon.png"), "")
            else:
                item_visible = QStandardItem(QIcon(":/images/resources/eye-icon-strike.png"), "")
            items.append(item_visible)
            # col 1 : image icon (for non-adjustment layer only) and name
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
                w, h = smallImg.width(), smallImg.height()
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
        layerview = mImg.getActiveLayer().view
        if layerview is not None:
            layerview.show()
            if TABBING:
                layerview.raise_()
        self.updateForm()
        for item in self.img.layersStack:
            if hasattr(item, 'sourceIndex'):
                combo = item.getGraphicsForm().sourceCombo
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

    def updateRows(self):
        """
        Update all rows.
        """
        count = self.model().rowCount()
        minInd, maxInd = self.model().index(0, 0), self.model().index(count-1, 3)
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
        # get target row and layer
        targetRow = self.indexAt(event.pos()).row()
        targetLayer = rStack[targetRow]
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
        for srcRow, tgtRow in sorted(rowMapping.items()):  # python 3 iteritems->items
            for col in range(0, colCount):
                # CAUTION : setItem calls the data changed event handler (cf. setLayers above)
                self.model().setItem(tgtRow, col, self.model().takeItem(srcRow, col))
        # remove moved rows from their initial place and keep track of moved items
        movedDict = rowMapping.copy()
        for row in reversed(sorted(rowMapping.keys())):  # python 3 iterkeys -> keys
            self.model().removeRow(row)
            for s, t in rowMapping.items():
                if t > row:
                    movedDict[s] -= 1
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
        if len(sel) > 1:
            self.currentWin.hide()
            self.currentWin = None
        elif len(sel) == 1:
            self.img.setActiveLayer(len(self.img.layersStack) - sel[0] - 1)
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
        # multiple selection : go to top of selection
        m = min(rows)
        if row != m:
            clickedIndex = self.model().index(m, clickedIndex.column())
        layer = self.img.layersStack[-1 - row]
        self.actionDup.setEnabled(not layer.isAdjustLayer())
        # toggle layer visibility
        if clickedIndex.column() == 0:
            # background layer is always visible
            if row == len(self.img.layersStack) - 1:
                return
            # layer.visible = not(layer.visible)
            layer.setVisible(not layer.visible)
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
                if i >= 0:
                    layer.parentImage.layersStack[i].applyToStack()
                else:
                    # top layer : update only the presentation layer
                    layer.parentImage.prLayer.execute(l=None, pool=None)
            self.img.onImageChanged()
        # update displayed window and active layer
        activeStackIndex = len(self.img.layersStack) - 1 - row
        activeLayer = self.img.setActiveLayer(activeStackIndex)
        # update color mask slider and label
        self.maskLabel.setEnabled(layer.maskIsSelected)
        self.maskSlider.setEnabled(activeLayer.maskIsSelected)
        self.maskValue.setEnabled(activeLayer.maskIsSelected)
        if self.currentWin is not None:
            if not self.currentWin.isFloating():
                # self.currentWin.hide()
                self.currentWin = None
        if hasattr(self.img.layersStack[activeStackIndex], "view"):
            self.currentWin = self.img.layersStack[activeStackIndex].view
        if self.currentWin is not None and activeLayer.visible:
            self.currentWin.show()
            self.currentWin.raise_()
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
        window.label.repaint()

    def initContextMenu(self):
        """
        Context menu initialization
        @return:
        @rtype: QMenu
        """
        menu = QMenu()
        # menu.actionReset = QAction('Reset To Default', None)
        # menu.actionLoadImage = QAction('Load New Image', None)
        # multiple selections
        menu.actionMerge = QAction('Merge Lower', None)
        # merge only adjustment layer with image layer
        menu.actionRepositionLayer = QAction('Reposition Layer(s)', None)
        menu.actionColorMaskEnable = QAction('Color', None)
        menu.actionOpacityMaskEnable = QAction('Opacity', None)
        menu.actionClippingMaskEnable = QAction('Clipping', None)
        menu.actionMaskDisable = QAction('Disabled', None)
        menu.actionMaskUndo = QAction('Undo Mask', None)
        menu.actionMaskRedo = QAction('Redo Mask', None)
        menu.actionMaskInvert = QAction('Invert Mask', None)
        menu.actionMaskReset_UM = QAction('Unmask All', None)
        menu.actionMaskReset_M = QAction('Mask All', None)
        menu.actionMaskCopy = QAction('Copy Mask to Clipboard', None)
        menu.actionImageCopy = QAction('Copy Image to Clipboard', None)
        menu.actionMaskPaste = QAction('Paste Mask', None)
        menu.actionImagePaste = QAction('Paste Image', None)
        menu.actionMaskDilate = QAction('Dilate Mask', None)
        menu.actionMaskErode = QAction('Erode Mask', None)
        menu.actionMaskSmooth = QAction('Smooth Mask', None)
        menu.actionMaskBright1 = QAction('Bright 1 Mask', None)
        menu.actionMaskBright2 = QAction('Bright 2 Mask', None)
        menu.actionMaskBright3 = QAction('Bright 3 Mask', None)
        menu.actionMaskDark1 = QAction('Dark 1 Mask', None)
        menu.actionMaskDark2 = QAction('Dark 2 Mask', None)
        menu.actionMaskDark3 = QAction('Dark 3 Mask', None)
        menu.actionMaskMid1 = QAction('Mid 1 Mask', None)
        menu.actionMaskMid2 = QAction('Mid 2 Mask', None)
        menu.actionMaskMid3 = QAction('Mid 3 Mask', None)
        menu.actionMergingFlag = QAction('Merged Layer', None)
        menu.actionMergingFlag.setCheckable(True)
        menu.actionColorMaskEnable.setCheckable(True)
        menu.actionOpacityMaskEnable.setCheckable(True)
        menu.actionClippingMaskEnable.setCheckable(True)
        menu.actionMaskDisable.setCheckable(True)
        ####################
        # Build menu
        ###################
        menu.addAction(menu.actionRepositionLayer)
        menu.addSeparator()
        # layer
        menu.addAction(menu.actionImageCopy)
        menu.addAction(menu.actionImagePaste)
        menu.addAction(menu.actionMergingFlag)
        menu.addSeparator()
        # mask
        menu.subMenuEnable = menu.addMenu('Mask...')
        menu.subMenuEnable.addAction(menu.actionColorMaskEnable)
        menu.subMenuEnable.addAction(menu.actionOpacityMaskEnable)
        menu.subMenuEnable.addAction(menu.actionClippingMaskEnable)
        menu.subMenuEnable.addAction(menu.actionMaskDisable)
        menu.addAction(menu.actionMaskUndo)
        menu.addAction(menu.actionMaskRedo)
        menu.addAction(menu.actionMaskInvert)
        menu.subMenuLum = menu.addMenu('Luminosity Mask...')
        for a in [menu.actionMaskBright1, menu.actionMaskBright2, menu.actionMaskBright3, menu.actionMaskDark1, menu.actionMaskDark2, menu.actionMaskDark3,
                  menu.actionMaskMid1, menu.actionMaskMid2, menu.actionMaskMid3]:
            menu.subMenuLum.addAction(a)
        menu.addAction(menu.actionMaskReset_UM)
        menu.addAction(menu.actionMaskReset_M)
        menu.addAction(menu.actionMaskCopy)
        menu.addAction(menu.actionMaskPaste)
        menu.addAction(menu.actionMaskDilate)
        menu.addAction(menu.actionMaskErode)
        menu.addAction(menu.actionMaskSmooth)
        menu.addSeparator()
        # miscellaneous
        # menu.addAction(menu.actionLoadImage)
        # to link actionDup with a shortcut,
        # it must be set in __init__
        menu.addAction(self.actionDup)
        menu.addAction(menu.actionMerge)
        # menu.addAction(menu.actionReset)
        return menu

    def contextMenuEvent(self, event):
        """
        context menu handler
        @param event
        @type event: QContextMenuEvent
        """
        selection = self.selectedIndexes()
        if not selection:
            return
        # get a fresh context menu without connected actions
        # and with state corresponding to the currently clicked layer
        self.cMenu = self.initContextMenu()
        # get current selection
        rows = set([mi.row() for mi in selection])
        rStack = self.img.layersStack[::-1]
        layers = [rStack[r] for r in rows]
        # get current position
        index = self.indexAt(event.pos())
        layerStackIndex = len(self.img.layersStack) - 1 - index.row()
        layer = self.img.layersStack[layerStackIndex]
        lowerVisible = self.img.layersStack[layer.getLowerVisibleStackIndex()]
        lower = self.img.layersStack[layerStackIndex - 1]  # case index == 0 doesn't matter
        # toggle actions
        self.cMenu.actionMergingFlag.setChecked(layer.mergingFlag)
        self.cMenu.actionMerge.setEnabled(not (hasattr(layer, 'inputImg') or hasattr(lowerVisible, 'inputImg')))
        self.actionDup.setEnabled(not layer.isAdjustLayer())
        self.cMenu.actionColorMaskEnable.setChecked(layer.maskIsSelected and layer.maskIsEnabled)
        self.cMenu.actionOpacityMaskEnable.setChecked((not layer.maskIsSelected) and layer.maskIsEnabled)
        self.cMenu.actionClippingMaskEnable.setChecked(layer.isClipping and (layer.maskIsSelected or layer.maskIsEnabled))
        self.cMenu.actionMaskDisable.setChecked(not(layer.isClipping or layer.maskIsSelected or layer.maskIsEnabled))
        self.cMenu.actionMaskUndo.setEnabled(layer.historyListMask.canUndo())
        self.cMenu.actionMaskRedo.setEnabled(layer.historyListMask.canRedo())
        self.cMenu.subMenuEnable.setEnabled(len(rows) == 1)
        self.cMenu.actionMaskPaste.setEnabled(not QApplication.clipboard().image().isNull())
        self.cMenu.actionImagePaste.setEnabled(not QApplication.clipboard().image().isNull())
        self.cMenu.actionMergingFlag.setEnabled(layer.isImageLayer())
        # Event handlers

        def RepositionLayer():
            layer.xOffset, layer.yOffset = 0, 0
            layer.Zoom_coeff = 1.0
            layer.AltZoom_coeff = 1.0
            layer.xAltOffset, layer.yAltOffset = 0, 0
            layer.updatePixmap()
            self.img.onImageChanged()

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
            self.maskLabel.setEnabled(layer.maskIsSelected)
            self.maskSlider.setEnabled(layer.maskIsSelected)
            self.maskValue.setEnabled(layer.maskIsSelected)
            layer.applyToStack()
            self.img.onImageChanged()

        def opacityMaskEnable():
            testUpperVisibility()
            layer.maskIsEnabled = True
            layer.maskIsSelected = False
            self.maskLabel.setEnabled(layer.maskIsSelected)
            self.maskSlider.setEnabled(layer.maskIsSelected)
            self.maskValue.setEnabled(layer.maskIsSelected)
            layer.applyToStack()
            self.img.onImageChanged()

        def clippingMaskEnable():
            layer.maskIsEnabled = True
            layer.maskIsSelected = False
            self.maskLabel.setEnabled(layer.maskIsSelected)
            self.maskSlider.setEnabled(layer.maskIsSelected)
            self.maskValue.setEnabled(layer.maskIsSelected)
            layer.isClipping = True
            layer.applyToStack()
            self.img.onImageChanged()

        def maskDisable():
            layer.maskIsEnabled = False
            layer.maskIsSelected = False
            self.maskLabel.setEnabled(layer.maskIsSelected)
            self.maskSlider.setEnabled(layer.maskIsSelected)
            self.maskValue.setEnabled(layer.maskIsSelected)
            layer.isClipping = False
            layer.applyToStack()
            self.img.onImageChanged()

        def undoMask():
            try:
                layer.mask = layer.historyListMask.undo(saveitem=layer.mask.copy())
                layer.applyToStack()
                self.img.onImageChanged()
            except ValueError:
                pass

        def redoMask():
            try:
                layer.mask = layer.historyListMask.redo()
                layer.applyToStack()
                self.img.onImageChanged()
            except ValueError:
                pass

        def maskInvert():
            layer.invertMask()
            # update mask stack
            layer.applyToStack()
            self.img.onImageChanged()

        def maskReset_UM():
            layer.resetMask(maskAll=False)
            # update mask stack
            for l in self.img.layersStack:
                l.updatePixmap(maskOnly=True)
            self.img.prLayer.execute(l=None, pool=None)
            self.img.onImageChanged()

        def maskReset_M():
            layer.resetMask(maskAll=True)
            # update mask stack
            for l in self.img.layersStack:
                l.updatePixmap(maskOnly=True)
            self.img.prLayer.execute(l=None, pool=None)
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
            self.img.prLayer.execute(l=None, pool=None)
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
            """
            Increase the masked part of the image
            """
            buf = QImageBuffer(layer.mask)
            buf[:, :, 2] = vImage.maskDilate(buf[:, :, 2])
            for l in self.img.layersStack:
                l.updatePixmap(maskOnly=True)
            self.img.prLayer.update()
            self.img.onImageChanged()

        def maskErode():
            """
            Reduce the masked part of the image
            """
            buf = QImageBuffer(layer.mask)
            buf[:, :, 2] = vImage.maskErode(buf[:, :, 2])
            for l in self.img.layersStack:
                l.updatePixmap(maskOnly=True)
            self.img.prLayer.update()
            self.img.onImageChanged()

        def maskSmooth():
            """
            Smooth the mask boundary
            """
            buf = QImageBuffer(layer.mask)
            buf[:, :, 2] = vImage.maskSmooth(buf[:, :, 2])
            for l in self.img.layersStack:
                l.updatePixmap(maskOnly=True)
            self.img.prLayer.update()
            self.img.onImageChanged()

        def maskBright1():
            layer.setMaskLuminosity(min=128, max=255)
            for l in self.img.layersStack:
                l.updatePixmap(maskOnly=True)
            self.img.prLayer.update()
            self.img.onImageChanged()

        def maskBright2():
            layer.setMaskLuminosity(min=192, max=255)
            for l in self.img.layersStack:
                l.updatePixmap(maskOnly=True)
            self.img.prLayer.update()
            self.img.onImageChanged()

        def maskBright3():
            layer.setMaskLuminosity(min=224, max=255)
            for l in self.img.layersStack:
                l.updatePixmap(maskOnly=True)
            self.img.prLayer.update()
            self.img.onImageChanged()

        def maskDark1():
            layer.setMaskLuminosity(min=0, max=128)

        def maskDark2():
            layer.setMaskLuminosity(min=0, max=64)
            for l in self.img.layersStack:
                l.updatePixmap(maskOnly=True)
            self.img.prLayer.update()
            self.img.onImageChanged()

        def maskDark3():
            layer.setMaskLuminosity(min=0, max=32)
            for l in self.img.layersStack:
                l.updatePixmap(maskOnly=True)
            self.img.prLayer.update()
            self.img.onImageChanged()

        def maskMid1():
            layer.setMaskLuminosity(min=64, max=192)
            for l in self.img.layersStack:
                l.updatePixmap(maskOnly=True)
            self.img.prLayer.update()
            self.img.onImageChanged()

        def maskMid2():
            layer.setMaskLuminosity(min=96, max=160)
            for l in self.img.layersStack:
                l.updatePixmap(maskOnly=True)
            self.img.prLayer.update()
            self.img.onImageChanged()

        def maskMid3():
            layer.setMaskLuminosity(min=112, max=144)
            for l in self.img.layersStack:
                l.updatePixmap(maskOnly=True)
            self.img.prLayer.update()
            self.img.onImageChanged()

        def mergingFlag(flag):
            layer.mergingFlag = flag

        self.cMenu.actionRepositionLayer.triggered.connect(RepositionLayer)
        # self.cMenu.actionLoadImage.triggered.connect(loadImage)
        self.cMenu.actionMerge.triggered.connect(merge)
        self.cMenu.actionColorMaskEnable.triggered.connect(colorMaskEnable)
        self.cMenu.actionOpacityMaskEnable.triggered.connect(opacityMaskEnable)
        self.cMenu.actionClippingMaskEnable.triggered.connect(clippingMaskEnable)
        self.cMenu.actionMaskDisable.triggered.connect(maskDisable)
        self.cMenu.actionMaskUndo.triggered.connect(undoMask)
        self.cMenu.actionMaskRedo.triggered.connect(redoMask)
        self.cMenu.actionMaskInvert.triggered.connect(maskInvert)
        self.cMenu.actionMaskReset_UM.triggered.connect(maskReset_UM)
        self.cMenu.actionMaskReset_M.triggered.connect(maskReset_M)
        self.cMenu.actionMaskCopy.triggered.connect(maskCopy)
        self.cMenu.actionMaskPaste.triggered.connect(maskPaste)
        self.cMenu.actionImageCopy.triggered.connect(imageCopy)
        self.cMenu.actionImagePaste.triggered.connect(imagePaste)
        self.cMenu.actionMaskDilate.triggered.connect(maskDilate)
        self.cMenu.actionMaskErode.triggered.connect(maskErode)
        self.cMenu.actionMaskSmooth.triggered.connect(maskSmooth)
        self.cMenu.actionMaskBright1.triggered.connect(maskBright1)
        self.cMenu.actionMaskBright2.triggered.connect(maskBright2)
        self.cMenu.actionMaskBright3.triggered.connect(maskBright3)
        self.cMenu.actionMaskDark1.triggered.connect(maskDark1)
        self.cMenu.actionMaskDark2.triggered.connect(maskDark2)
        self.cMenu.actionMaskDark3.triggered.connect(maskDark3)
        self.cMenu.actionMaskMid1.triggered.connect(maskMid1)
        self.cMenu.actionMaskMid2.triggered.connect(maskMid2)
        self.cMenu.actionMaskMid3.triggered.connect(maskMid3)
        self.cMenu.actionMergingFlag.toggled.connect(mergingFlag)
        self.cMenu.exec_(event.globalPos() - QPoint(400, 0))
        # update table
        for row in rows:
            self.updateRow(row)

