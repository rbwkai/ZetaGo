"""Loading and caching of image / sound assets for the GUI."""

import os

import pygame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "assets", "img")
SND = os.path.join(ROOT, "assets", "sound")
ICO = os.path.join(ROOT, "assets", "icons")


class Assets:
    def __init__(self):
        self._black_raw = pygame.image.load(os.path.join(IMG, "black.png")).convert_alpha()
        self._white_raw = pygame.image.load(os.path.join(IMG, "white.png")).convert_alpha()
        self._wood_raw = pygame.image.load(os.path.join(IMG, "kaya.jpg")).convert()
        self.icon = self._try(lambda: pygame.image.load(
            os.path.join(ICO, "apple-touch-icon-180x180.png")).convert_alpha())
        self.place_sound = self._try(lambda: pygame.mixer.Sound(
            os.path.join(SND, "zz-un-floor-goban.v7.mp3")))
        if self.place_sound is not None:
            self.place_sound.set_volume(0.5)
        self.muted = False
        self._stone_cache = {}
        self._wood_cache = {}

    @staticmethod
    def _try(fn):
        try:
            return fn()
        except Exception:
            return None

    def stone(self, color, diameter):
        key = (color, diameter)
        if key not in self._stone_cache:
            raw = self._black_raw if color == 1 else self._white_raw
            self._stone_cache[key] = pygame.transform.smoothscale(raw, (diameter, diameter))
        return self._stone_cache[key]

    def wood(self, size):
        if size not in self._wood_cache:
            self._wood_cache[size] = pygame.transform.smoothscale(self._wood_raw, (size, size))
        return self._wood_cache[size]

    def play_place(self):
        if self.place_sound is not None and not self.muted:
            try:
                self.place_sound.play()
            except Exception:
                pass
