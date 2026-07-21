"""Renders the goban and maps between pixels and board coordinates (row 0 = top)."""

import pygame

from . import theme

# Star points (hoshi) per board size.
_HOSHI = {
    5: [(2, 2)],
    7: [(3, 3), (2, 2), (2, 4), (4, 2), (4, 4)],
    9: [(4, 4), (2, 2), (2, 6), (6, 2), (6, 6)],
}
_COLS = "ABCDEFGHJKLMNOPQRST"


class BoardView:
    def __init__(self, rect, n, assets):
        self.rect = pygame.Rect(rect)
        self.n = n
        self.assets = assets
        avail = min(self.rect.w, self.rect.h)
        self.size = avail
        self.cell = avail / n
        self.radius = int(self.cell * 0.46)
        self.board_x = self.rect.x + (self.rect.w - avail) // 2
        self.board_y = self.rect.y + (self.rect.h - avail) // 2
        self.ox = self.board_x + self.cell / 2
        self.oy = self.board_y + self.cell / 2

    def xy(self, row, col):
        return (int(self.ox + col * self.cell), int(self.oy + row * self.cell))

    def cell_at(self, pos):
        px, py = pos
        c = round((px - self.ox) / self.cell)
        r = round((py - self.oy) / self.cell)
        if 0 <= r < self.n and 0 <= c < self.n:
            cx, cy = self.xy(r, c)
            if (px - cx) ** 2 + (py - cy) ** 2 <= (self.cell * 0.5) ** 2:
                return (r, c)
        return None

    def draw(self, surf, board, hover=None, hover_color=1):
        surf.blit(self.assets.wood(self.size), (self.board_x, self.board_y))
        x0, y0 = self.xy(0, 0)
        x1, y1 = self.xy(self.n - 1, self.n - 1)
        for i in range(self.n):
            x = int(self.ox + i * self.cell)
            y = int(self.oy + i * self.cell)
            pygame.draw.line(surf, theme.GRID_LINE, (x0, y), (x1, y), 2)
            pygame.draw.line(surf, theme.GRID_LINE, (x, y0), (x, y1), 2)
        for (r, c) in _HOSHI.get(self.n, []):
            pygame.draw.circle(surf, theme.HOSHI, self.xy(r, c), max(3, int(self.cell * 0.06)))
        self._labels(surf)

        for r in range(self.n):
            for c in range(self.n):
                bit = 1 << (r * self.n + c)
                if board.black & bit:
                    self._stone(surf, r, c, 1)
                elif board.white & bit:
                    self._stone(surf, r, c, -1)

        if board.last_move is not None:
            cx, cy = self.xy(*divmod(board.last_move, self.n))
            pygame.draw.circle(surf, theme.LASTMOVE, (cx, cy), max(4, int(self.cell * 0.12)), 3)

        if hover is not None:
            ghost = self.assets.stone(hover_color, self.radius * 2).copy()
            ghost.set_alpha(theme.GHOST_ALPHA)
            cx, cy = self.xy(*hover)
            surf.blit(ghost, (cx - self.radius, cy - self.radius))

    def _stone(self, surf, r, c, color):
        cx, cy = self.xy(r, c)
        rad = self.radius
        shadow = pygame.Surface((rad * 2 + 6, rad * 2 + 6), pygame.SRCALPHA)
        pygame.draw.circle(shadow, theme.SHADOW, (rad + 3, rad + 3), rad)
        surf.blit(shadow, (cx - rad - 1, cy - rad + 2))
        surf.blit(self.assets.stone(color, rad * 2), (cx - rad, cy - rad))

    def _labels(self, surf):
        f = theme.font(15)
        for c in range(self.n):
            x, _ = self.xy(0, c)
            t = f.render(_COLS[c], True, theme.GRID_LINE)
            surf.blit(t, t.get_rect(center=(x, int(self.board_y + self.size - self.cell * 0.20))))
        for r in range(self.n):
            _, y = self.xy(r, 0)
            t = f.render(str(self.n - r), True, theme.GRID_LINE)
            surf.blit(t, t.get_rect(center=(int(self.board_x + self.cell * 0.20), y)))
