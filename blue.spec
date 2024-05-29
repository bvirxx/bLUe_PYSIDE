# -*- mode: python ; coding: utf-8 -*-

imageformats = [('.\\venv6_123_2\\Lib\site-packages\\PySide6\plugins\\imageformats', 'plugins\\imageformats')]
platforms = [('.\\venv6_123_2\\Lib\site-packages\\PySide6\\plugins\\platforms', 'plugins\\platforms')]

data_1 = [('bLUeNN\\pretrained_models\\sRGB\\*', 'bLUeNN\\pretrained_models\\sRGB')]
data_2 = [('blue.ui', '.'), ('README.md', '.'), ('LICENSE.TXT', '.'), ('logo.png', '.'),
          ('config_win.json', '.'), ('brushes\README.TXT', 'brushes'), ('bLUe.qss', '.')]

a = Analysis(
    ['blue.py'],
    pathex=[],
    binaries=[('C:\\standalone\\exiftool(-k).exe', 'bin')] + imageformats + platforms,
    datas=data_1 + data_2,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='blue',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory='.',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='blue',
)
