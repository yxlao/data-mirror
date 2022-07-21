# nerfmirror

Unofficial data downloader for common NeRF datasets.

## Usage

```
nerfmirror.py [-h] [--download_dir DOWNLOAD_DIR] [--cache_dir CACHE_DIR] {dataset_name}
```

Examples:
```shell
# NeRF LLFF dataset, save to "./data" by default.
python nerfmirror.py nerf_synthetic

# NeRF LLFF dataset, save to a specified dir.
python nerfmirror.py --download_dir=data nerf_synthetic

# NeRF LLFF dataset, with cache dir for uncompressed data.
# If the cache_dir is specified:
#   - The cached files will be checked and reused if the checksum matches.
#   - Extraction will be performed.
#   - Any additional files in the extraction directory will be unchanged.
python nerfmirror.py --cache_dir=/mnt/data/nerfmirror nerf_synthetic
python nerfmirror.py --download_dir=data --cache_dir=/mnt/data/nerfmirror nerf_synthetic
```

## Datasets

- `nerf_synthetic`
  ```text
  ${download_dir}/nerf_synthetic
  ├── chair
  ├── drums
  ├── ficus
  ├── hotdog
  ├── lego
  ├── materials
  ├── mic
  ├── README.txt
  └── ship
  ```
  This comes from the [original NeRF paper](https://github.com/bmild/nerf).
  License: [link](https://github.com/bmild/nerf/blob/master/LICENSE).
- `nerf_llff`
  ```text
  ${download_dir}/nerf_llff_data
  ├── fern
  ├── flower
  ├── fortress
  ├── horns
  ├── leaves
  ├── orchids
  ├── room
  └── trex
  ```
  This is the [pre-processed dataset](https://github.com/bmild/nerf) of
  the [original LLFF dataset](https://github.com/fyusion/llff).
  License: [link](https://github.com/Fyusion/LLFF/blob/master/LICENSE),
  [link](https://github.com/bmild/nerf/blob/master/LICENSE).


## License

The code in this repository is under the MIT license. You must accept the
original authors' licenses before using this repository to download datasets.
