# Dataset plan

- `datasets/deepcrack`: image/mask pairs for segmentation and coverage labels.
- `datasets/sdnet2018`: cracked/noncracked images for diversity and no-crack class.
- `datasets/crack_segmentation`: deferred for now because the archive is approximately 2 GB.

## Current two-dataset plan

Use DeepCrack and SDNET2018 initially.

- DeepCrack image-mask pairs provide minor/moderate/severe crack-coverage proxy labels.
- SDNET2018 noncracked images provide the no-crack class.
- SDNET2018 cracked images without masks are retained as `severity_pending`; they are not used for four-class severity training yet. They can later be used for binary crack/no-crack pretraining.

To avoid overwhelming the small masked dataset, use a documented, reproducible sample of SDNET2018 noncracked images for the first four-class experiment. Do not discard the full source data.

Do not assign minor/moderate/severe to a cracked image without a mask or expert label. Keep original source and scene identity to prevent crop leakage.
