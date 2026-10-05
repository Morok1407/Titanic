# Build with: python -m PyInstaller --noconfirm Titanic.spec
from pathlib import Path

root = Path(SPECPATH)
a = Analysis(
    [str(root / "launcher.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[
        (str(root / "web"), "web"),
        (str(root / "models" / "titanic_mlp.npz"), "models"),
        (str(root / "reports" / "metrics.json"), "reports"),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="Titanic", debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=True,
    icon=str(root / "build_assets" / "titanic.ico"),
)
