# -*- mode: python ; coding: utf-8 -*-
# PyInstaller build spec for OCR-Excel

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# RapidOCR ONNX 모델 파일 자동 수집
rapidocr_datas = collect_data_files('rapidocr_onnxruntime')
easyocr_datas = collect_data_files('easyocr')
paddleocr_datas = collect_data_files('paddleocr')
google_genai_hiddenimports = collect_submodules('google.genai')


def collect_tree_data(src_dir, dest_root):
    datas = []
    if not os.path.isdir(src_dir):
        return datas
    for root, _dirs, files in os.walk(src_dir):
        if not files:
            continue
        rel = os.path.relpath(root, src_dir)
        dest = dest_root if rel == '.' else os.path.join(dest_root, rel)
        for name in files:
            datas.append((os.path.join(root, name), dest))
    return datas


easyocr_model_datas = collect_tree_data(os.path.expanduser(r'~\.EasyOCR'), '.EasyOCR')
paddleocr_model_datas = collect_tree_data(os.path.expanduser(r'~\.paddleocr'), '.paddleocr')
paddlex_model_datas = collect_tree_data(
    os.path.expanduser(r'~\.paddlex\official_models'),
    os.path.join('.paddlex', 'official_models'),
)

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        *rapidocr_datas,
        *easyocr_datas,
        *paddleocr_datas,
        *easyocr_model_datas,
        *paddleocr_model_datas,
        *paddlex_model_datas,
        # 한국어 OCR 모델 + 딕셔너리 + config.json
        ('models/korean_rec.onnx', 'models'),
        ('models/korean_dict.txt', 'models'),
        ('models/korean_config.yaml', 'models'),
        ('config.json', '.'),
    ],
    hiddenimports=[
        'rapidocr_onnxruntime',
        'onnxruntime',
        'pdfplumber',
        'pdfplumber.page',
        'pdfplumber.utils',
        'pdfplumber.table',
        'openpyxl',
        'cv2',
        'numpy',
        'pandas',
        'fitz',
        'PIL',
        'groq',
        'google',
        'google.genai',
        'google.genai.types',
        'easyocr',
        'paddleocr',
        'scipy',
        'skimage',
        'httpx',
        'httpcore',
        'anyio',
        'sniffio',
        'pydantic',
        'pydantic_core',
        'core.runtime_flags',
        *collect_submodules('rapidocr_onnxruntime'),
        *collect_submodules('pdfplumber'),
        *google_genai_hiddenimports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'IPython',
        'notebook',
        'pytest',
        'img2table',
        'numba',
        'polars',
        'paddle.audio',
        'paddle.dataset',
        'paddle.distributed',
        'paddle.distribution',
        'paddle.geometric',
        'paddle.incubate',
        'paddle.jit.sot',
        'paddle.onnx',
        'paddle.profiler',
        'paddle.quantization',
        'paddle.text',
        'paddle.vision',
        'torch.distributed',
        'torch.utils.tensorboard',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OCR-Excel',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI 모드 (콘솔 창 없음)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OCR-Excel',
)
