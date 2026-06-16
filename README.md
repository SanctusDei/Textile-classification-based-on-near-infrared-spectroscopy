# Textile Classification Dataset — NIR Spectroscopy

Near-infrared (NIR) spectroscopy dataset for textile fiber classification, collected using a **DLP NIRScan nano** spectrometer.

## Dataset Overview

| Property | Value |
|----------|-------|
| **Samples** | 188 pure-fiber spectra |
| **Spectral range** | 900–1700 nm |
| **Spectral bands** | 228 pixels |
| **Classes** | 6 |
| **Instrument** | DLP NIRScan nano |
| **Additional data** | 40 blend spectra, 132 fabric images |

## Classes

| Class | Count | Swatches |
|-------|-------|----------|
| Polyester (PET) | 83 | 24 |
| Nylon (PA) | 45 | 4 |
| Cotton | 45 | 4 |
| Wool | 5 | 5 |
| Acrylic | 5 | 5 |
| Acetate | 5 | 5 |

## Data Format

Each CSV file contains a 22-line metadata header followed by 228 rows of spectral data:

- `Wavelength (nm)` — wavelength in nanometers
- `Absorbance (AU)` — absorbance in absorbance units
- `Reference Signal (unitless)` — background reference
- `Sample Signal (unitless)` — raw sample signal

### Acquisition Parameters

| Parameter | Value |
|-----------|-------|
| Exposure time | 0.635 ms |
| Repeated scans | 6 |
| PGA Gain | 16 |
| Total measurement time | 2.451 s / sample |

## Directory Structure

```
data/
├── csv/                        # Main dataset — 188 pure-fiber spectra
├── preprocessing/
│   ├── Pure/                   # Reorganized pure-fiber spectra
│   │   ├── 01_Acetate/         # 5 spectra
│   │   ├── 02_Cotton/          # 45 spectra + 4 images
│   │   ├── 03_Nylon_PA/        # 45 spectra + 4 images
│   │   ├── 04_Polyester_PET/   # 83 spectra + 24 images
│   │   ├── 05_Acrylic/         # 5 spectra
│   │   ├── 06_Wool/            # 5 spectra
│   │   └── 07_Background/      # 10 PVC background spectra
│   ├── Blends/                 # 40 blend-fabric spectra + 5 images
│   │   ├── Blend_Cotton55_Polyester45/
│   │   ├── Blend_Cotton80_Polyester20/
│   │   ├── Blend_Nylon30_Polyester70/
│   │   └── Blend_Wool35_Polyester65/
│   └── total/                  # 35 additional fabric images
└── raw/
    └── image/                  # 58 raw fabric images
```

## Citation

If you use this dataset, please cite:
```
@dataset{textile-nir-classification,
  title     = {NIR Spectroscopy Dataset for Textile Fiber Classification},
  author   = {Your Name},
  year     = {2026},
  url      = {https://github.com/SanctusDei/Textile-classification-based-on-near-infrared-spectroscopy}
}
```

## License

This dataset is provided for research purposes.
