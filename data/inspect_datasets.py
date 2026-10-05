"""Report dataset file counts and image/mask dimensions before any training."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
IMAGE_EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
for p in [ROOT/'datasets/deepcrack',ROOT/'datasets/sdnet2018',ROOT/'datasets/crack_segmentation']:
    print(f'\n{p.name}')
    if p.exists():
        for d in sorted(x for x in p.rglob('*') if x.is_dir()):
            files=[x for x in d.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTS]
            print(f'  {d.relative_to(p)}: {len(files)} image files')
