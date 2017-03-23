# -*- mode: python -*-

# *** CAUTION : pyinstaller bLUe.py will erase modifications in this file ****
# TO BUILD a distribution, run pyinstaller bLUe.spec
block_cipher = None

a = Analysis(['bLUe.py'],
             pathex=['D:\\PycharmProject\\CS4'],
             binaries=[('D:\\Python\\Lib\\cv2.pyd', '.'), ('D:\\Python\\Lib\\tbb.dll', '.'), ('C:\standalone\\exiftool\\exiftool(-k)', '.') ],
             datas=[('essai1.ui', '.'), ('resources_rc.pyc', '.'), ('qsettingsexample.ini', '.')],
             hiddenimports=['layerView'],
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
          debug=True,  # False
          strip=False,
          upx=True,
          console=True )  # False
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='bLUe')
