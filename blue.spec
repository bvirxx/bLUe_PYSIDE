# -*- mode: python -*-

block_cipher = None

imageformats = [('D:\\Python36\\Lib\site-packages\\PySide2\plugins\\imageformats', 'plugins\\imageformats')]
platforms = [('D:\\Python36\\Lib\site-packages\\PySide2\\plugins\\platforms', 'plugins\\platforms')]


a = Analysis(['bLUe.py'],
             pathex=['D:\\PycharmProject\\CS4_PYSIDE'],
             binaries=[('C:\\standalone\\exiftool\\exiftool.exe', 'bin')] + imageformats + platforms,
             datas=[('blue.ui', '.'), ('README.TXT', '.'), ('LICENSE.TXT', '.'), ('logo.png', '.'), ('logo.ico', '.'), ('config.json', '.')],
             hiddenimports=['PySide2.QtXml', 'pywt._extensions._cwt'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.Qt', 'PIL.ImageQt', 'PySide2.QtQuick', 'PySide2.QtWebEngineWidgets'],  # the last 2 modules are added for qmake failure to find qml install
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
