from pathlib import Path
import urllib
import urllib.request
import os
import hashlib
from tqdm import tqdm
import zipfile
import shutil
import argparse
import contextlib
import tempfile

_dataset_registry = dict()


@contextlib.contextmanager
def _make_dir_or_temp_dir(cache_dir):
    """
    Args:
        cache_dir: Path to the cache directory.
            - If cache_dir is not None, this function will attempt to create
                cache_dir. The cache_dir will not be deleted once the context
                exits.
            - If cache_dir is None, a temporary cache_dir will be created.
                The entire cache_dir will be deleted once the context exits.
    """
    if cache_dir:
        cache_dir = Path(cache_dir)
        if cache_dir.is_file():
            raise ValueError(
                f"Cache dir {cache_dir} already exists and it is a file.")
        else:
            cache_dir.mkdir(exist_ok=True, parents=True)
        is_temp_dir = False
    else:
        cache_dir = Path(tempfile.mkdtemp())
        is_temp_dir = True

    try:
        yield cache_dir
    finally:
        if is_temp_dir:
            shutil.rmtree(cache_dir)


def _lookahead(iterable):
    """
    Yields (item, has_more) for an iterable.

    Ref:https://stackoverflow.com/a/1630350/1255535
    """
    it = iter(iterable)
    last = next(it)
    for val in it:
        yield last, True
        last = val
    yield last, False


class RegisterDataset(object):

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def __call__(self, cls):
        _dataset_registry[self.dataset_name] = cls

        class wrapped_cls(cls):
            cls.dataset_name = self.dataset_name

        return wrapped_cls


class Dataset:

    def __init__(self, download_dir, cache_dir):
        self.download_dir = Path(download_dir).resolve()
        if cache_dir is None:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = Path(cache_dir).resolve()

    def download(self):
        raise NotImplementedError("Abstract method not implemented.")

    @staticmethod
    def download_url(url, output_path):

        class DownloadProgressBar(tqdm):

            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with DownloadProgressBar(unit="B",
                                 unit_scale=True,
                                 miniters=1,
                                 desc=url.split("/")[-1]) as t:
            urllib.request.urlretrieve(url,
                                       filename=output_path,
                                       reporthook=t.update_to)

    @staticmethod
    def sha256sum(file_path):
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"{file_path} not found.")

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256.update(byte_block)
        return sha256.hexdigest()

    @staticmethod
    def download_single_file(
        url,
        sha256,
        byproduct_dir_name,
        delete_dir_name,
        download_dir,
        cache_dir,
    ):
        """
        Args:
            url: URL of the file to download.
            sha256: SHA256 hash of the file.
            byproduct_dir_name: The relative directory inside download_dir after
                extraction for sanity check. This must be a string and be
                relative to download_dir.
            delete_dir_name: Sometimes, some extracted dirs (e.g. "__MACOSX")
                are not needed. Specify this to delete them after extraction.
                This must be a string and be relative to download_dir.
            download_dir: Download root dir. A directory within this directory
                will be created. E.g. {download_dir}/nerf_llff.
            cache_dir: Cache directory. If cache_dir is not None, the original
                download file will be cached in cache_dir.

        """

        with _make_dir_or_temp_dir(cache_dir) as cache_dir:
            # Wrap dirs.
            download_dir = Path(download_dir).resolve()
            cache_dir = Path(cache_dir).resolve()
            if not isinstance(byproduct_dir_name, str):
                raise ValueError("byproduct_dir_name must be a string.")

            # Print dirs.
            print(f"download_dir: {download_dir}")
            print(f"cache_dir: {cache_dir}")

            # Check cache.
            file_name = os.path.basename(urllib.parse.urlparse(url).path)
            file_path = cache_dir / file_name
            sha256_valid = False
            if file_path.is_file():
                print(f"{file_path} exists, checking checksum.")
                sha256_valid = Dataset.sha256sum(file_path) == sha256
                if sha256_valid:
                    print(f"{file_path} checksum matches, skipping download.")
                else:
                    print(
                        f"{file_path} checksum mismatches, will download again."
                    )

            # Download.
            if not sha256_valid:
                Dataset.download_url(url, file_path)
                sha256_valid = Dataset.sha256sum(file_path) == sha256
                if sha256_valid:
                    print(f"{file_path} downloaded successfully.")
                else:
                    raise ValueError(
                        f"{file_path} checksum mismatch."
                        f"Expected checksum {sha256}. "
                        f"Please download {url} to {cache_dir} manually.")

            # Extract.
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(download_dir)

            # Check byproducts.
            byproduct_dir = (download_dir / byproduct_dir_name).resolve()
            if not byproduct_dir.exists():
                raise FileNotFoundError(
                    f"byproduct_dir {byproduct_dir} not found after extraction."
                )

            # Delete delete_dir_name.
            if delete_dir_name is not None:
                delete_dir = (download_dir / delete_dir_name).resolve()
                if delete_dir.exists():
                    shutil.rmtree(delete_dir)

            # List directory of byproduct_dir.
            print(f"Extracted:")
            print(f"{byproduct_dir}")
            for f, has_more in _lookahead(byproduct_dir.iterdir()):
                if has_more:
                    print(f"├── {f.name}")
                else:
                    print(f"└── {f.name}")


@RegisterDataset("nerf_synthetic")
class NeRFSynthetic(Dataset):
    """
    Args:
        download_dir: The parent directory to save the dataset.

    The resulting folder structure:
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
    """

    def __init__(self, download_dir, cache_dir):
        super().__init__(download_dir, cache_dir)

    def download(self):
        Dataset.download_single_file(
            url=
            "https://github.com/yxlao/nerfmirror/releases/download/20220618/nerf_synthetic.zip",
            sha256=
            "f01fd1b4ab045b0d453917346f26f898657bb5bec4834b95fdad1f361826e45e",
            byproduct_dir_name="nerf_synthetic",
            delete_dir_name="__MACOSX",
            download_dir=self.download_dir,
            cache_dir=self.cache_dir,
        )


@RegisterDataset("nerf_llff")
class NeRFLLFF(Dataset):
    """
    Args:
        download_dir: The parent directory to save the dataset.

    The resulting folder structure:
        ${download_dir}/nerf_llff_data
        ├── fern
        ├── flower
        ├── fortress
        ├── horns
        ├── leaves
        ├── orchids
        ├── room
        └── trex
    """

    def __init__(self, download_dir, cache_dir):
        super().__init__(download_dir, cache_dir)

    def download(self):
        Dataset.download_single_file(
            url=
            "https://github.com/yxlao/nerfmirror/releases/download/20220618/nerf_llff_data.zip",
            sha256=
            "5794b432feaf4f25bcd603addc6ad0270cec588fed6a364b7952001f07466635",
            byproduct_dir_name="nerf_llff_data",
            delete_dir_name=None,
            download_dir=self.download_dir,
            cache_dir=self.cache_dir,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Datamirror: downloader for common NeRF datasets.")
    parser.add_argument("dataset_name",
                        type=str,
                        help="Dataset name",
                        choices=sorted(list(_dataset_registry.keys())))
    parser.add_argument("--download_dir",
                        dest="download_dir",
                        default="data",
                        help="Download directory")
    parser.add_argument("--cache_dir",
                        dest="cache_dir",
                        default=None,
                        help="Cache directory")
    args = parser.parse_args()

    dataset_class = _dataset_registry[args.dataset_name](
        download_dir=args.download_dir,
        cache_dir=args.cache_dir,
    )
    dataset_class.download()


if __name__ == "__main__":
    main()
