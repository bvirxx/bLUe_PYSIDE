# -*- mode: python -*-

block_cipher = None

a = Analysis(['bLUe.py'],
             pathex=['D:\\PycharmProject\\CS4_PYSIDE'],
             binaries=[('C:\\standalone\\exiftool\\exiftool.exe', 'bin')],
             datas=[('blue.ui', '.'), ('settings.py', '.')],
             hiddenimports=['PySide2.QtXml', 'pywt._extensions._cwt'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
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
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='bLUe')
