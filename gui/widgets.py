"""Minimal clickable button widget."""

import pygame

from . import theme


class Button:
    def __init__(self, rect, label, callback, kind="normal", selected=False, hint=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.callback = callback
        self.kind = kind          # "normal" | "accent" | "warn"
        self.enabled = True
        self.selected = selected
        self.hover = False
        self.hint = hint          # optional keyboard-shortcut hint, e.g. "P"

    def handle(self, event):
        if not self.enabled:
            return
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()

    def draw(self, surf):
        base = {"accent": theme.ACCENT, "warn": theme.WARN}.get(self.kind, theme.PANEL_LINE)
        if not self.enabled:
            color, txt = (44, 47, 54), theme.TEXT_DIM
        elif self.selected:
            color, txt = theme.ACCENT, (14, 16, 20)
        else:
            color = tuple(min(255, c + (22 if self.hover else 0)) for c in base)
            txt = (14, 16, 20) if self.kind != "normal" else theme.TEXT
        pygame.draw.rect(surf, color, self.rect, border_radius=10)
        if self.selected or self.hover:
            pygame.draw.rect(surf, theme.ACCENT, self.rect, width=2, border_radius=10)
        label = theme.font(20, bold=True).render(self.label, True, txt)
        surf.blit(label, label.get_rect(center=self.rect.center))
        if self.hint and self.enabled:
            hint = theme.font(13, bold=True).render(self.hint, True, txt)
            hint.set_alpha(150)
            surf.blit(hint, hint.get_rect(topright=(self.rect.right - 7, self.rect.top + 5)))
