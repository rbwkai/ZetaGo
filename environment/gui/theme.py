"""Visual theme: window size, colours, and fonts for the ZetaGo GUI."""

import pygame

# Window / layout
WINDOW_W, WINDOW_H = 1000, 720
FPS = 60
PANEL_W = 300

# Colours
BG = (22, 24, 28)
PANEL_BG = (32, 35, 41)
PANEL_CARD = (40, 44, 51)
PANEL_LINE = (56, 60, 69)
GRID_LINE = (38, 28, 14)
HOSHI = (28, 20, 10)
TEXT = (234, 236, 240)
TEXT_DIM = (152, 158, 168)
TEXT_FAINT = (108, 114, 124)
ACCENT = (92, 162, 232)
GOOD = (122, 202, 132)
WARN = (226, 120, 92)
LASTMOVE = (228, 70, 70)
BLACK_DOT = (18, 20, 24)
WHITE_DOT = (238, 240, 244)
SHADOW = (0, 0, 0, 95)
GHOST_ALPHA = 115

_FONTS = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _FONTS:
        name = pygame.font.match_font("dejavusans,arial,helvetica,sans")
        f = pygame.font.Font(name, size) if name else pygame.font.SysFont(None, size)
        f.set_bold(bold)
        _FONTS[key] = f
    return _FONTS[key]
