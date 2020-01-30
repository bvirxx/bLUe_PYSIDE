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
from tempfile import mktemp

from PySide2.QtCore import Qt, QDir
from os.path import isfile

from PySide2.QtWidgets import QMessageBox, QPushButton, QFileDialog, QDialog, QSlider, QVBoxLayout, QHBoxLayout, QLabel, \
    QCheckBox, QFormLayout, QLineEdit, QDialogButtonBox
from bLUeTop.utils import QbLUeSlider

##################
# file extension constants

IMAGE_FILE_EXTENSIONS = (".jpg", ".JPG", ".png", ".PNG", ".tif", ".TIF", ".bmp", ".BMP")
RAW_FILE_EXTENSIONS = (".nef", ".NEF", ".dng", ".DNG", ".cr2", ".CR2")
IMAGE_FILE_NAME_FILTER = ['Image Files (*.jpg *.png *.tif *.JPG *.PNG *.TIF)']
#################


class dimsInputDialog(QDialog):
    """
    Simple input Dialog for image width and height.
    """
    def __init__(self, dims, keepBox=False):
        """

        @param dims: {'w': width, 'h': height}
        @type dims: dict
        """
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle('Image Dimensions')
        self.dims = dims
        self.r = 1.0
        if dims['w'] > 0:
            self.r = dims['h'] / dims['w']
        fLayout = QFormLayout()
        self.fields = []
        self.checkBox = None
        for i in range(2):
            lineEdit = QLineEdit()
            label = "Width (px)" if i == 0 else "Height (px)"
            lineEdit.setText(str(dims['w']) if i == 0 else str(dims['h']))
            fLayout.addRow(label, lineEdit)
            self.fields.append(lineEdit)
            lineEdit.textEdited.connect(self.keepRatioW if i == 0 else self.keepRatioH)
        if keepBox:
            label1 = "Keep Aspect Ratio"
            self.checkBox = QCheckBox()
            fLayout.addRow(label1, self.checkBox)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok, Qt.Horizontal)
        fLayout.addRow(buttonBox)
        self.setLayout(fLayout)
        buttonBox.accepted.connect(self.accept)

    def keepRatioH(self, text):
        if self.checkBox is None:
            return
        elif not self.checkBox.isChecked():
            return
        try:
            w1, h1 = int(self.fields[0].text()), int(self.fields[1].text())
        except ValueError:
            return
        w1 = int(h1 / self.r)
        self.fields[0].setText(str(w1))

    def keepRatioW(self, text):
        if self.checkBox is None:
            return
        elif not self.checkBox.isChecked():
            return
        try:
            w1, h1 = int(self.fields[0].text()), int(self.fields[1].text())
        except ValueError:
            return
        h1 = int(w1 * self.r)
        self.fields[1].setText(str(h1))

    def accept(self):
        """
        button slot
        """
        try:
            self.dims['w'] = int(self.fields[0].text())
            self.dims['h'] = int(self.fields[1].text())
        except ValueError:
            return
        super().accept()


def dlgInfo(text, info=''):
    """
    Shows a simple information dialog.
    @param text:
    @type text: str
    @param info:
    @type info: str
    """
    msg = QMessageBox()
    msg.setWindowTitle('Information')
    msg.setIcon(QMessageBox.Information)
    msg.setText(text)
    msg.setInformativeText(info)
    msg.exec_()


def dlgWarn(text, info=''):
    """
    Shows a simple warning dialog.
    @param text:
    @type text: str
    @param info:
    @type info: str
    """
    msg = QMessageBox()
    msg.setWindowTitle('Warning')
    msg.setIcon(QMessageBox.Warning)
    msg.setText(text)
    msg.setInformativeText(info)
    msg.exec_()


def saveChangeDialog(img):
    """
    Save/discard dialog. Returns the chosen button.
    @param img: image to save
    @type img: vImage
    @return:
    @rtype: QMessageBox.StandardButton
    """
    reply = QMessageBox()
    reply.setText("%s was modified" % img.meta.name if len(img.meta.name) > 0 else 'unnamed image')
    reply.setInformativeText("Save your changes ?")
    reply.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
    reply.setDefaultButton(QMessageBox.Save)
    ret = reply.exec_()
    return ret


class savingDialog(QDialog):
    """
    File dialog with options.
    We use a standard QFileDialog as a child widget and we
    forward its methods to the top level.
    """
    def __init__(self, parent, text, lastDir):
        """

        @param parent:
        @type parent: QObject
        @param text:
        @type text: str
        @param lastDir:
        @type lastDir:str
        """
        # QDialog __init__
        super().__init__()
        self.setWindowTitle(text)
        # File Dialog
        self.dlg = QFileDialog(caption=text, directory=lastDir)
        self.metaOption = QCheckBox('Remove Meta')
        # sliders
        self.sliderComp = QbLUeSlider(Qt.Horizontal)
        self.sliderComp.setTickPosition(QSlider.TicksBelow)
        self.sliderComp.setRange(0, 9)
        self.sliderComp.setSingleStep(1)
        self.sliderComp.setValue(5)
        self.sliderQual = QbLUeSlider(Qt.Horizontal)
        self.sliderQual.setTickPosition(QSlider.TicksBelow)
        self.sliderQual.setRange(0, 100)
        self.sliderQual.setSingleStep(10)
        self.sliderQual.setValue(90)
        self.dlg.setVisible(True)
        l = QVBoxLayout()
        h = QHBoxLayout()
        l.addWidget(self.dlg)
        h.addWidget(self.metaOption)
        h.addWidget(QLabel("Quality"))
        h.addWidget(self.sliderQual)
        h.addWidget(QLabel("Compression"))
        h.addWidget(self.sliderComp)
        l.addLayout(h)
        self.setLayout(l)
        # file dialog close event handler

        def f():
            self.close()
        self.dlg.finished.connect(f)

    def exec_(self):
        # QDialog exec_
        super().exec_()
        # forward file dialog result
        return self.dlg.result()

    def selectFile(self, fileName):
        self.dlg.selectFile(fileName)

    def selectedFiles(self):
        return self.dlg.selectedFiles()

    def directory(self):
        return self.dlg.directory()


def saveDlg(img, mainWidget):
    """
    Image saving dialogs. The actual saving is
    done by a call to mImage.save(). Metadata is copied from sidecar
    to image file. The function returns the image file name.
    Exception ValueError or IOError are raised if the saving fails.
    @param img:
    @type img: vImage
    @param mainWidget:
    @type mainWidget: QWidget
    @return: filename
    @rtype: str
    """
    # get last accessed dir
    lastDir = str(mainWidget.settings.value("paths/dlgdir", QDir.currentPath()))
    # file dialogs
    dlg = savingDialog(mainWidget, "Save", lastDir)
    # default saving format JPG
    dlg.selectFile(img.filename[:-3] + 'JPG')
    dlg.dlg.currentChanged.connect(lambda f: print(f))
    if dlg.exec_():
        newDir = dlg.directory().absolutePath()
        mainWidget.settings.setValue('paths/dlgdir', newDir)
        filenames = dlg.selectedFiles()
        if filenames:
            filename = filenames[0]
        else:
            raise ValueError("You must select a file")
        if isfile(filename):
            reply = QMessageBox()
            reply.setWindowTitle('Warning')
            reply.setIcon(QMessageBox.Warning)
            reply.setText("File %s already exists\n" % filename)
            reply.setStandardButtons(QMessageBox.Cancel)
            accButton = QPushButton("Save as New Copy")
            rejButton = QPushButton("OverWrite")
            reply.addButton(accButton, QMessageBox.AcceptRole)
            reply.addButton(rejButton, QMessageBox.RejectRole)
            reply.setDefaultButton(accButton)
            reply.exec_()
            retButton = reply.clickedButton()
            # build a unique name
            if retButton is accButton:
                i = 0
                base = filename
                if '_copy' in base:
                    flag = '_'
                else:
                    flag = '_copy'
                while isfile(filename):
                    filename = base[:-4] + flag + str(i) + base[-4:]
                    i = i+1
            # overwrite
            elif retButton is rejButton:
                pass
            else:
                raise ValueError("Saving Operation Failure")
        # get parameters
        quality = dlg.sliderQual.value()
        compression = dlg.sliderComp.value()
        # call mImage.save to write image to file and return a thumbnail
        # throw ValueError or IOError
        thumb = img.save(filename, quality=quality, compression=compression)
        # write metadata
        if not dlg.metaOption.isChecked():
            tempFilename = mktemp('.jpg')
            # save thumb jpg to temp file
            thumb.save(tempFilename)
            # copy temp file to image file
            img.restoreMeta(img.filename, filename, thumbfile=tempFilename)
            os.remove(tempFilename)
        return filename
    else:
        raise ValueError("Saving Operation Failure")


def openDlg(mainWidget, ask=True, multiple=False):
    """
    if multiple is true returns a list of file names,
     otherwise returns a file name or None.
    @param mainWidget:
    @type mainWidget:
    @param ask:
    @type ask:
    @param multiple:
    @type multiple: boolean
    @return:
    @rtype: string or list of strings
    """
    if ask and mainWidget.label.img.isModified:
        ret = saveChangeDialog(mainWidget.label.img)
        if ret == QMessageBox.Yes:
            try:
                saveDlg(mainWidget.label.img, mainWidget)
            except (ValueError, IOError) as e:
                dlgWarn(str(e))
                return
        elif ret == QMessageBox.Cancel:
            return
    # don't ask again for saving
    mainWidget.label.img.isModified = False
    lastDir = str(mainWidget.settings.value('paths/dlgdir', '.'))
    if multiple:
        # allow multiple selections
        filenames = QFileDialog.getOpenFileNames(mainWidget, "select", lastDir, " *".join(IMAGE_FILE_EXTENSIONS) + " *".join(RAW_FILE_EXTENSIONS))
        return filenames[0]
    # select a single file
    dlg = QFileDialog(mainWidget, "select", lastDir, " *".join(IMAGE_FILE_EXTENSIONS) + " *".join(RAW_FILE_EXTENSIONS))
    if dlg.exec_():
        filenames = dlg.selectedFiles()
        newDir = dlg.directory().absolutePath()
        mainWidget.settings.setValue('paths/dlgdir', newDir)
        return filenames[0]
    return None
