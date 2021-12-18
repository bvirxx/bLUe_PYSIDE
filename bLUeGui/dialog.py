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
import textwrap
from os.path import basename
from PySide2.QtCore import Qt, QDir, QSize

from PySide2.QtWidgets import QMessageBox, QPushButton, QFileDialog, QDialog, QSlider, QVBoxLayout, QHBoxLayout, QLabel, \
    QCheckBox, QFormLayout, QLineEdit, QDialogButtonBox, QScrollArea, QProgressDialog

from bLUeTop.utils import QbLUeSlider

##################
# file extension constants
BLUE_FILE_EXTENSIONS = (".blu", ".BLU", ".bLU")
IMAGE_FILE_EXTENSIONS = (".jpg", ".JPG", ".png", ".PNG", ".tif", ".TIF", ".bmp", ".BMP")
RAW_FILE_EXTENSIONS = (".nef", ".NEF", ".dng", ".DNG", ".cr2", ".CR2")
IMAGE_FILE_NAME_FILTER = ['Image Files (*.jpg *.png *.tif *.JPG *.PNG *.TIF)']


#################


class dimsInputDialog(QDialog):
    """
    Simple input Dialog for image width and height.
    """

    def __init__(self, w, h, keepAspectRatio=True, keepBox=False):
        """

        @param dims: {'w': width, 'h': height, 'kr': bool}
        @type dims: dict
        """
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle('Image Dimensions')
        self.dims = {'w': w, 'h': h, 'kr': keepAspectRatio}
        self.r = 1.0
        if w > 0:
            self.r = h / w
        fLayout = QFormLayout()
        self.fields = []
        self.checkBox = None
        for i in range(2):
            lineEdit = QLineEdit()
            label = "Width (px)" if i == 0 else "Height (px)"
            lineEdit.setText(str(w) if i == 0 else str(h))
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
        self.onAccept = lambda: None
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
            if self.checkBox is not None:
                self.dims['kr'] = self.checkBox.isChecked()
        except ValueError:
            return
        super().accept()
        self.onAccept()


def dlgInfo(text, info='', parent=None):
    """
    Shows a simple information dialog.
    @param parent:
    @type parent: QWidget
    @param text:
    @type text: str
    @param info:
    @type info: str
    """
    msg = QMessageBox(parent=parent)
    msg.setWindowTitle('Information')
    msg.setIcon(QMessageBox.Information)
    msg.setText(text)
    msg.setInformativeText(info)
    msg.exec_()


def dlgWarn(text, info='', parent=None):
    """
    Shows a simple warning dialog.
    @param parent:
    @type parent: QWidget
    @param text:
    @type text: str
    @param info:
    @type info: str
    """
    msg = QMessageBox(parent=parent)
    msg.setWindowTitle('Warning')
    msg.setIcon(QMessageBox.Warning)
    msg.setText(text)
    msg.setInformativeText(info)
    msg.exec_()


def workInProgress(parent=None):
    """
    Inits a modal QProgressBar
    @param parent:
    @type parent:
    @return:
    @rtype: QProgressDialog
    """
    progress = QProgressDialog(parent=parent)
    progress.setFixedSize(200, 60)
    progress.setStyleSheet("""background-color: rgb(20,20,100);
                              color: rgb(220,220,220);""")
    progress.setWindowModality(Qt.ApplicationModal)
    progress.setAttribute(Qt.WA_DeleteOnClose)
    progress.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint)
    progress.setCancelButtonText('')
    return progress


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
        self.dlg.setOption(QFileDialog.DontUseNativeDialog)
        self.metaOption = QCheckBox('Remove Meta')
        # sliders
        self.sliderComp = QbLUeSlider(Qt.Horizontal)
        self.sliderComp.setTickPosition(QSlider.TicksBelow)
        self.sliderComp.setRange(0, 9)
        self.sliderComp.setSingleStep(1)
        self.sliderComp.setValue(3)  # 3 = default opencv imwrite value
        self.sliderQual = QbLUeSlider(Qt.Horizontal)
        self.sliderQual.setTickPosition(QSlider.TicksBelow)
        self.sliderQual.setRange(0, 100)
        self.sliderQual.setSingleStep(10)
        self.sliderQual.setValue(95)  # 95 = default opencv imwrite value
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


class labelDlg(QDialog):
    """
    Displays a floating modal text window.
    If search is True, a search field editor is added on top of the window.
    """

    def __init__(self, parent=None, title='', wSize=QSize(500, 500), scroll=True, search=False, modal=True):
        super().__init__(parent)
        self.setWindowTitle(parent.tr(title))
        self.setStyleSheet(" * {background-color: rgb(220, 220, 220); color: black;}\
                            QLabel {selection-background-color: blue; selection-color: white}\
                            QLineEdit {background-color: white;}")
        self.setModal(modal)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignTop)
        # self.label.setStyleSheet("selection-background-color: blue; selection-color: white}")
        vl = QVBoxLayout()
        if search:
            ed = QLineEdit()
            ed.setMaximumWidth(300)
            ed.setPlaceholderText('Search')
            vl = QVBoxLayout()
            hl = QHBoxLayout()
            hl.addWidget(ed)
            button = QPushButton('Next')
            button.setAutoDefault(False);
            button.setMaximumWidth(60)
            hl.addWidget(button)
            vl.addLayout(hl)
            matches = []
            current = 0

            def f(searchedText):
                import re
                nonlocal current
                matches.clear()
                current = 0
                matches.extend([m.span() for m in
                                re.finditer(searchedText, self.label.text(), re.IGNORECASE | re.MULTILINE | re.DOTALL)])
                if matches:
                    item = matches[0]
                    self.label.setSelection(item[0], item[1] - item[0])
                    metrics = self.label.fontMetrics()
                    tabSize = 4
                    rect = metrics.boundingRect(0, 0, 150000, 150000, self.label.alignment() | Qt.TextExpandTabs,
                                                self.label.text()[:item[1]], tabSize)
                    scarea.ensureVisible(0, rect.height())

            def g():
                nonlocal current
                if not matches or not button.isDown():
                    return
                current = (current + 1) % len(matches)
                item = matches[current]
                self.label.setSelection(item[0], item[1] - item[0])
                metrics = self.label.fontMetrics()
                tabSize = 4
                rect = metrics.boundingRect(0, 0, 150000, 150000, self.label.alignment() | Qt.TextExpandTabs,
                                            self.label.text()[:item[1]], tabSize)
                scarea.ensureVisible(0, rect.height())

            button.pressed.connect(g)
            ed.textEdited.connect(f)

        if scroll:
            scarea = QScrollArea()
            scarea.setWidget(self.label)
            scarea.setWidgetResizable(True)
            vl.addWidget(scarea)
        else:
            vl.addWidget(self.label)
        self.setLayout(vl)
        self.setFixedSize(wSize)

    def wrapped(self, s):
        """
        Returns wrapped text, according to the current font and size of label.
        NOT updated when these parameters are modified.
        @param s: text to wrap
        @type s: str
        @return:
        @rtype: list of str
        """
        metrics = self.label.fontMetrics()
        tabSize = 4
        # get max character count per line
        testText = 'WWWWWWWWWWWWWWW'  # start from a minimum width !
        while metrics.boundingRect(0, 0, 150000, 150000, self.label.alignment() | Qt.TextExpandTabs,
                                   testText, tabSize).width() < self.label.width():
            testText += 'W'
        # wrap text while keeping existing newlines
        s = '\n'.join(['\n'.join(textwrap.wrap(line, len(testText), break_long_words=False, replace_whitespace=False))
                       for line in s.splitlines()])
        return s


def saveDlg(img, mainWidget, ext='jpg', selected=True):
    """
    Image saving dialog.
    If selected is False, initially the filename box is left empty and no file is selected.
    A ValueError exception is raised if the dialog is aborted.
    @param img:
    @type img: vImage
    @param mainWidget:
    @type mainWidget: QWidget
    @param selected:
    @type selected: boolean
    @return: filename, quality, compression, metaOption
    @rtype: str, int, int, boolean
    """
    # get last accessed dir
    lastDir = str(mainWidget.settings.value("paths/dlgsavedir", QDir.currentPath()))
    # file dialogs
    dlg = savingDialog(mainWidget, "Save", lastDir)
    if selected:
        # default saving format jpg
        dlg.selectFile(basename(img.filename)[:-3] + ext)
    filename = ''
    if dlg.exec_():
        newDir = dlg.directory().absolutePath()
        mainWidget.settings.setValue('paths/dlgsavedir', newDir)
        filenames = dlg.selectedFiles()
        if filenames:
            filename = filenames[0]
    else:
        raise ValueError("You must select a file")
    return filename, dlg.sliderQual.value(), dlg.sliderComp.value(), not dlg.metaOption.isChecked()


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
    filter = "Images ( *" + " *".join(IMAGE_FILE_EXTENSIONS) + " *" + " *".join(RAW_FILE_EXTENSIONS) + " *".join(
        BLUE_FILE_EXTENSIONS) + ")"
    if multiple:
        # allow multiple selections
        filenames = QFileDialog.getOpenFileNames(mainWidget, "select", lastDir, filter)
        return filenames[0]
    # select a single file
    dlg = QFileDialog(mainWidget, "select", lastDir, filter)
    if dlg.exec_():
        filenames = dlg.selectedFiles()
        newDir = dlg.directory().absolutePath()
        mainWidget.settings.setValue('paths/dlgdir', newDir)
        return filenames[0]
    return None
