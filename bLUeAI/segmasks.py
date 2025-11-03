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
import io
import base64
import json
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QImage
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFileDialog, QPlainTextEdit, QHBoxLayout, QApplication
from PIL import Image, ImageDraw
from google.genai.errors import APIError

from bLUeGui.bLUeImage import QImageBuffer
from bLUeGui.dialog import dlgWarn
from bLUeGui.memory import weakProxy
from bLUeTop.Gui import window
import bLUeTop.settings
from bLUeTop.utils import QbLUePushButton
from bLUeTop.versatileImg import vImage


class GeminiError(Exception):
    pass
    """
    def __init__(self, *args):
        # Call the base class constructor with the parameters it needs
        super().__init__(*args)
    """

def parse_json(json_output: str):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i + 1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output


def extract_segmentation_masks(aitool):
    img = aitool.inputImg
    if bLUeTop.settings.HAS_GENAI:
        from google import genai
        from google.genai import types
    else:
        dlgWarn('google-genai is not installed')
        return

    # Load image
    im = Image.fromarray(QImageBuffer(img)[:, :, [2, 1, 0, 3]])
    #im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    prompt = aitool.promptEdit.toPlainText()

    config = types.GenerateContentConfig(
        # set thinking_budget to 0 for better results in object detection
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    try:
        with genai.Client() as client:
            response = client.models.generate_content(
                model="gemini-2.5-flash",  #flash-image
                contents=[prompt, im],  # Pillow images can be directly passed as inputs (which will be converted by the SDK)
                config=config
            )
    except (APIError, ValueError) as e:
        raise GeminiError(str(e))

    console = aitool.console
    maskList = []
    # Parse JSON response
    try:
        items = json.loads(parse_json(response.text))
    except json.JSONDecodeError as e:
        console.appendPlainText(response.text)
        return
    if items:
        console.appendPlainText(f'{len(items)} Mask(s) found:')
    else:
        console.appendPlainText('No mask found')
    # Process each mask
    for i, item in enumerate(items):
        # Get bounding box coordinates
        box = item["box_2d"]
        y0 = int(box[0] / 1000 * im.size[1])
        x0 = int(box[1] / 1000 * im.size[0])
        y1 = int(box[2] / 1000 * im.size[1])
        x1 = int(box[3] / 1000 * im.size[0])

        # Skip invalid boxes
        if y0 >= y1 or x0 >= x1:
            continue

        # Process mask
        png_str = item["mask"]
        if not png_str.startswith("data:image/png;base64,"):
            continue

        # Remove prefix
        png_str = png_str.removeprefix("data:image/png;base64,")
        mask_data = base64.b64decode(png_str)
        mask_ori = Image.open(io.BytesIO(mask_data))

        # Resize mask to match bounding box
        mask = mask_ori.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR )

        # Convert mask to numpy array for processing
        mask_array = np.array(mask)

        # Create bLUe mask from this
        overlay = Image.new('RGBA', im.size, vImage.defaultColor_UnMasked.getRgb())
        overlay_draw = ImageDraw.Draw(overlay)

        color = vImage.defaultColor_Masked.getRgb()
        for y in range(y0, y1):
            for x in range(x0, x1):
                if mask_array[y - y0, x - x0] > 127:  # confidence threshold for mask
                    overlay_draw.point((x, y), fill=color)

        # Create and save overlay
        #mask_filename = f"{item['label']}_{i}_mask.png"
        overlay_filename = f"{item['label']}_{i}.png"
        composite = Image.alpha_composite(im.convert('RGBA'), overlay)
        compositePath = os.path.join(aitool.dir_path, overlay_filename)
        composite.save(compositePath)
        #console.appendHtml(f"<span style='color: #FF0000'>{aitool.dir_path}\\{overlay_filename}</span>")
        console.appendHtml(f"<span style='color: #FF0000'>{overlay_filename}</span>")
        maskList.append(compositePath)
    console.appendPlainText(f'mask(s) size {im.size}')
    return maskList


class AIForm(QWidget):

    default_prompt = """
          Give the segmentation masks for the objects. Slightly smooth mask edges to avoid jagged edges.
          Output a JSON list of segmentation masks where each entry contains the 2D
          bounding box in the key "box_2d", the segmentation mask in key "mask", and
          the text label in the key "label". Use descriptive labels.
          """
    default_title = "AI Tool "

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dir_path = window.settings.value('paths/gendir', '')
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.Tool)
        self.setWindowTitle("AI Tool")
        self.resize(600, 500)
        self.inputImg = None
        self.promptEdit = QPlainTextEdit()
        self.promptEdit.setFixedHeight(150)
        self.promptEdit.setPlainText(self.default_prompt)
        self.outputLabel = QLabel(f"Output Directory: {self.dir_path}")
        self.console = QPlainTextEdit()
        self.console.setReadOnly = True
        self.outputBrowseButton = QbLUePushButton("Browse output directory")
        self.outputBrowseButton.clicked.connect(self.browseOutputDir)
        self.resetPromptBtn = QbLUePushButton('Reset Prompt to Default')
        self.resetPromptBtn.clicked.connect(self.resetPrompt)
        self.runButton = QbLUePushButton("  Run ")
        self.runButton.clicked.connect(self.runSegmentation)
        self.runButton.setEnabled(bLUeTop.settings.HAS_GENAI)
        self.importButton = QbLUePushButton("Import Masks")
        self.importButton.clicked.connect(self.importMasks)
        self.importButton.setEnabled(False)
        self.maskList = []

        layout = QVBoxLayout()
        #layout.addWidget(self.imagePathLabel)
        #layout.addWidget(self.imagePathEdit)
        hlay1 = QHBoxLayout()
        hlay1.addWidget(QLabel('Gemini Prompt Editor'))
        hlay1.addStretch(1)
        hlay1.addWidget(self.resetPromptBtn)
        layout.addLayout(hlay1)
        layout.addWidget(self.promptEdit)
        hlay2 = QHBoxLayout()
        hlay2.addWidget(self.outputLabel)
        hlay2.addStretch(1)
        hlay2.addWidget(self.outputBrowseButton)
        layout.addLayout(hlay2)
        layout.addWidget(self.console)
        hlay3 = QHBoxLayout()
        hlay3.addStretch(1)
        hlay3.addWidget(self.runButton)
        hlay3.addWidget(self.importButton)
        layout.addLayout(hlay3)

        self.setLayout(layout)
        self.setVisible(False)
        self.setWhatsThis(
            """
            <b>Conversational mask generation</b><br>
            Edit the Gemini prompt, if needed.<br>
            To <b>generate the masks</b>, click the <i>Run</i> button.<br>
            <i>Note.</i> If the <i>Run</i> button is disabled, verify that the package google-genai is installed
            and that the <i>GEMINI_API_KEY</i> environment variable is set.<br><br>
            To <b>import all masks</b> into the current layer, 
            press the <i>Import Masks</i> button.<br>
            To <b>import a subset of masks</b>, right-click the layer name
            in the <i>Layer View</i> (right pane), and choose <i>Import Mask</i>
            from the context menu which opens. Next, use the <i>file explorer</i> to select
            one or more masks to import.<br>.
            """
        )
    """
    def clear(self):
        #self.outputLabel.setText("Output Dir: ")
        self.inputImg = None
    """

    def resetPrompt(self):
        self.promptEdit.setPlainText(self.default_prompt)

    def setInput(self, im):
        self.inputImg = weakProxy(im)
        self.inputPath = im.filename
        self.setWindowTitle(self.default_title + self.inputPath)

    def browseOutputDir(self):
        self.dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if self.dir_path:
            window.settings.setValue('paths/gendir', self.dir_path)
            self.outputLabel.setText("Output Dir " + self.dir_path)

    def runSegmentation(self):
        if not self.inputImg:
            self.console.appendHtml("<span style='color: #FF0000'>No input image</span>")
            return
        if not self.dir_path:
            self.console.appendHtml("<span style='color: #FF0000'>No output directory</span>")
            return
        self.importButton.setEnabled(False)
        self.console.appendPlainText('Current Prompt (Use Prompt Editor above to edit)')
        # copy current prompt to console
        self.console.appendPlainText(self.promptEdit.toPlainText())
        self.console.appendPlainText('Processing prompt...')
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()
            self.maskList = extract_segmentation_masks(self)
            self.importButton.setEnabled(True)
        except GeminiError as e:
            QApplication.restoreOverrideCursor()
            QApplication.processEvents()
            dlgWarn('Gemini error', info=str(e))
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.processEvents()

    def importMasks(self):
        layer = self.inputImg.getActiveLayer()
        if self.maskList:
            #newDir = Path(maskList[0]).parent
            #Gui.window.settings.setValue(key, newDir)
            qp = QPainter(layer.mask)
            qp.setCompositionMode(QPainter.CompositionMode.CompositionMode_Darken)
            count = 0
            for f in self.maskList:
                m = QImage(f)
                if m.isNull():
                    self.console.appendPlainText(f'{f} is not a valid mask')
                else:
                    qp.drawImage(layer.mask.rect(), m)
                    count += 1
            self.console.appendPlainText(f'{count} mask(s) imported into {layer.name} layer')
            self.inputImg.prLayer.execute(l=None, pool=None)
            self.inputImg.onImageChanged()