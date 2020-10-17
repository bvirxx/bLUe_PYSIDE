# -*- mode: python -*-

block_cipher = None

imageformats = [('C:\Program Files\Python37\Lib\site-packages\\PySide2\plugins\\imageformats', 'plugins\\imageformats')]
platforms = [('C:\Program Files\Python37\Lib\site-packages\\PySide2\\plugins\\platforms', 'plugins\\platforms')]
numpy_dlls = [('C:\Program Files\\Python37\\Lib\\site-packages\\numpy\\dlls\\*.dll', '.')]

a = Analysis(['bLUe.py'],
             pathex=['D:\\PycharmProject\\CS4_PYSIDE'],
             binaries=[('C:\\standalone\\exiftool(-k).exe', 'bin')] + imageformats + platforms + numpy_dlls,
             datas=[('blue.ui', '.'), ('README.md', '.'), ('LICENSE.TXT', '.'), ('logo.png', '.'), ('logo.ico', '.'), ('config_win.json', '.'), ('brushes\README.TXT', 'brushes')],
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
