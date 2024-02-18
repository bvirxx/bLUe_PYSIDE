# -*- mode: python -*-

import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

block_cipher = None

imageformats = [('.\\venv6_108_3\\Lib\site-packages\\PySide6\plugins\\imageformats', 'plugins\\imageformats')]
platforms = [('.\\venv6_108_3\\Lib\site-packages\\PySide6\\plugins\\platforms', 'plugins\\platforms')]

data_1 = [('bLUeNN\\pretrained_models\\sRGB\\*', 'bLUeNN\\pretrained_models\\sRGB')]
data_2 = [('blue.ui', '.'), ('README.md', '.'), ('LICENSE.TXT', '.'), ('logo.png', '.'),
          ('config_win.json', '.'), ('brushes\README.TXT', 'brushes'), ('bLUe.qss', '.')]

a = Analysis(['bLUe.py'],
             pathex=[('C:\\Windows\\System32\\downlevel')],
             binaries=[('C:\\standalone\\exiftool(-k).exe', 'bin')] + imageformats + platforms,
             datas=data_1 + data_2,
             hiddenimports=['PySide6.QtXml', 'pywt._extensions._cwt'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.Qt', 'PIL.ImageQt', 'PySide6.QtQuick', 'PySide6.QtWebEngineWidgets'],  # the last 2 modules are added because of qmake failure finding qml install
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='bLUe',
          debug=False,
          strip=False,
          upx=True,
          icon='logo.ico',
          console=False )

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='bLUe')
