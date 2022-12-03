# -*- mode: python -*-

block_cipher = None

imageformats = [('.\\venv\\Lib\site-packages\\PySide2\plugins\\imageformats', 'plugins\\imageformats')]
platforms = [('.\\venv\\\\Lib\site-packages\\PySide2\\plugins\\platforms', 'plugins\\platforms')]

data_0 = [(HOMEPATH + '\\Pyside2\\*.dll', 'Pyside2')]  #  workaround for PyInstaller hooks problem (Pyside2)
data_1 = [('bLUeNN\\pretrained_models\\sRGB\\*', 'bLUeNN\\pretrained_models\\sRGB')]
data_2 = [('blue.ui', '.'), ('README.md', '.'), ('LICENSE.TXT', '.'), ('logo.png', '.'), ('logo.ico', '.'),
          ('config_win.json', '.'), ('brushes\README.TXT', 'brushes'), ('bLUe.qss', '.')]

a = Analysis(['bLUe.py'],
             pathex=[('C:\\Windows\\System32\\downlevel')],
             binaries=[('C:\\standalone\\exiftool(-k).exe', 'bin')] + imageformats + platforms,
             datas=data_0 + data_1 + data_2,
             hiddenimports=['PySide2.QtXml', 'pywt._extensions._cwt'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.Qt', 'PIL.ImageQt', 'PySide2.QtQuick', 'PySide2.QtWebEngineWidgets'],  # the last 2 modules are added because of qmake failure finding qml install
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
