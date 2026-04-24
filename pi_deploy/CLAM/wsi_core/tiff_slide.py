"""
tiff_slide.py
A minimal openslide-compatible wrapper for single-level TIFF/OME-TIFF files.
Mimics the openslide.OpenSlide API used by CLAM's WholeSlideImage.
"""
import numpy as np
from PIL import Image
import tifffile


class TiffSlide:
    """Drop-in replacement for openslide.OpenSlide for single-level TIFFs."""

    def __init__(self, path: str):
        self._path = path
        self._img = tifffile.imread(path)  # shape: (H, W, 3) or (H, W, 4)

        # ensure RGB uint8
        if self._img.ndim == 2:
            self._img = np.stack([self._img] * 3, axis=-1)
        if self._img.shape[2] == 4:
            self._img = self._img[:, :, :3]
        if self._img.dtype != np.uint8:
            self._img = (self._img / self._img.max() * 255).astype(np.uint8)

        h, w = self._img.shape[:2]

        # build a small pyramid by successive 2x downsamples (openslide-style)
        self._levels = [self._img]
        while w > 512 or h > 512:
            w, h = max(1, w // 2), max(1, h // 2)
            self._levels.append(self._img[::len(self._levels)*2, ::len(self._levels)*2])

        # recompute properly
        self._levels = [self._img]
        factor = 1
        while True:
            factor *= 2
            dh, dw = self._img.shape[0] // factor, self._img.shape[1] // factor
            if dh < 64 or dw < 64:
                break
            self._levels.append(self._img[::factor, ::factor])

        self.level_count = len(self._levels)
        self.level_dimensions = tuple(
            (lvl.shape[1], lvl.shape[0]) for lvl in self._levels
        )
        self.level_downsamples = tuple(
            float(self._img.shape[1]) / lvl.shape[1] for lvl in self._levels
        )
        self.properties = {
            "openslide.vendor": "tiffslide",
            "tiffslide.path": path,
        }

    def read_region(self, location, level, size):
        """Returns a PIL RGBA image, matching openslide behaviour."""
        x, y = location
        w, h = size
        ds = int(self.level_downsamples[level])

        # convert level-0 coords to this level's coords
        lx = x // ds
        ly = y // ds

        lvl_img = self._levels[level]
        lh, lw = lvl_img.shape[:2]

        # clamp
        x1, y1 = min(lx, lw), min(ly, lh)
        x2, y2 = min(lx + w, lw), min(ly + h, lh)

        patch = lvl_img[y1:y2, x1:x2]

        # pad if near edge
        if patch.shape[0] < h or patch.shape[1] < w:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            canvas[:patch.shape[0], :patch.shape[1]] = patch
            patch = canvas

        # openslide returns RGBA
        rgba = np.dstack([patch, np.full(patch.shape[:2], 255, dtype=np.uint8)])
        return Image.fromarray(rgba, "RGBA")

    def get_best_level_for_downsample(self, downsample: float) -> int:
        for i, ds in enumerate(self.level_downsamples):
            if ds > downsample:
                return max(0, i - 1)
        return len(self._levels) - 1

    def get_thumbnail(self, size):
        img = Image.fromarray(self._img)
        img.thumbnail(size, Image.LANCZOS)
        return img

    def close(self):
        pass  # nothing to close

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
