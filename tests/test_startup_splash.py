from __future__ import annotations

import unittest

from PIL import Image, ImageOps

import app


class TestStartupSplash(unittest.TestCase):
    def test_supplied_background_is_used_and_static_bar_is_removed(self):
        self.assertTrue(app.SPLASH_IMAGE.is_file())
        with Image.open(app.SPLASH_IMAGE) as source:
            original_size = source.size
            source_pixel = source.convert("RGB").getpixel((600, 707))
            cleaned = app._remove_embedded_bar(source)

        self.assertEqual(original_size, app.IMAGE_SIZE)
        self.assertGreater(source_pixel[2], source_pixel[0] * 2)
        self.assertLess(cleaned.getpixel((600, 707))[2], 80)

    def test_background_fits_window_and_loading_line_is_centered(self):
        with Image.open(app.SPLASH_IMAGE) as source:
            fitted = ImageOps.fit(
                app._remove_embedded_bar(source),
                (1920, 1080),
                method=Image.Resampling.LANCZOS,
            )
            left, right, y = app._bar_geometry(source.size, (1920, 1080))

        self.assertEqual(fitted.size, (1920, 1080))
        self.assertAlmostEqual((left + right) / 2, 960, delta=1)
        self.assertGreater(y, 540)
        self.assertLess(y, 1080)


if __name__ == "__main__":
    unittest.main()
