"""ZetaGo Pygame GUI: human vs human / random bot / KataGo, on the bitboard engine."""

import os
import random
import threading

import pygame

from engine import GoBoard, BLACK, WHITE
from katago_gtp import KataGoConfig, KataGoGTP
from . import theme
from .assets import Assets
from .board_view import BoardView
from .widgets import Button

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KATAGO_EXE = os.path.join(ROOT, "katago", "bin", "katago")
# Dedicated config whose rules match the engine (positional/area/no-tax/suicide-illegal).
KATAGO_CFG = os.path.join(ROOT, "katago", "configs", "gui_gtp.cfg")

OPP_HUMAN, OPP_RANDOM, OPP_KATAGO = "Human", "Random", "KataGo"
OPPONENTS = (OPP_HUMAN, OPP_RANDOM, OPP_KATAGO)

_COLS = "ABCDEFGHJKLMNOPQRST"  # Go columns skip 'I'; matches the board coordinate labels


def _find_model():
    mdir = os.path.join(ROOT, "katago", "models")
    if os.path.isdir(mdir):
        for suffix in (".txt.gz", ".bin.gz", ".bin"):
            for f in sorted(os.listdir(mdir)):
                if f.endswith(suffix):
                    return os.path.join(mdir, f)
    return ""


class GoGUI:
    def __init__(self, board_size=7, fast=False):
        pygame.init()
        try:
            pygame.mixer.init()
        except Exception:
            pass
        self.n = board_size
        self.fast = fast            # skip the random-bot move delay (used by selftest)
        self.screen = pygame.display.set_mode((theme.WINDOW_W, theme.WINDOW_H))
        pygame.display.set_caption("ZetaGo — 7x7 Go")
        self.clock = pygame.time.Clock()
        self.assets = Assets()
        if self.assets.icon:
            pygame.display.set_icon(self.assets.icon)

        self.panel_x = theme.WINDOW_W - theme.PANEL_W
        self.view = BoardView((0, 0, self.panel_x, theme.WINDOW_H), self.n, self.assets)

        self.opponent = OPP_RANDOM
        self.human_color = BLACK
        self.board = GoBoard(self.n)
        self.moves = []             # list of (player, point|None)
        self.katago = None
        self.katago_thinking = False
        self._pending = None        # (row, col, raw) from the KataGo worker thread
        self._bot_timer = None
        self._status = ""
        self._winner_text = ""
        self._score_text = ""
        self._confirm_resign = False
        self.hover = None
        self.buttons = []
        self.running = False
        self.mode = "menu"
        self._build_menu()

    # ---------------------------------------------------------------- KataGo
    def _start_katago(self):
        cfg = KataGoConfig(
            executable=KATAGO_EXE, model_path=_find_model(), config_path=KATAGO_CFG,
            board_size=self.n, komi=self.board.komi,
        )
        self.katago = KataGoGTP(cfg)
        self.katago.start()

    def _stop_katago(self):
        if self.katago is not None:
            try:
                self.katago.close()
            except Exception:
                pass
            self.katago = None

    def _katago_think(self, color):
        try:
            r, c, raw = self.katago.genmove(color)
            self._pending = (r, c, raw)
        except Exception as exc:
            self._pending = (None, None, f"error:{exc}")

    # ------------------------------------------------------------- game flow
    def _begin_game(self):
        self.board = GoBoard(self.n)
        self.moves = []
        self._bot_timer = None
        self._pending = None
        self.katago_thinking = False
        self._status = ""
        self._winner_text = ""
        self._score_text = ""
        self._confirm_resign = False
        self.hover = None
        self._stop_katago()
        if self.opponent == OPP_KATAGO:
            try:
                self._start_katago()
            except Exception as exc:
                self.opponent = OPP_RANDOM
                self._status = f"KataGo unavailable, using Random ({exc})"
        self.mode = "play"
        self._build_play_buttons()

    def _is_human_turn(self):
        if self.board.is_game_over():
            return False
        if self.opponent == OPP_HUMAN:
            return True
        return self.board.current_player == self.human_color

    def _cancel_resign(self):
        if self._confirm_resign:
            self._confirm_resign = False
            self._status = ""

    def _do_move(self, row, col, sync_katago):
        player = self.board.current_player
        if not self.board.play_move(row, col):
            return False
        self._cancel_resign()
        self.moves.append((player, (row, col)))
        self.assets.play_place()
        if self.katago is not None and sync_katago:
            try:
                self.katago.play(player, row, col)
            except Exception:
                pass
        self._bot_timer = None
        return True

    def _do_pass(self, sync_katago):
        player = self.board.current_player
        self._cancel_resign()
        self.board.pass_move()
        self.moves.append((player, None))
        if self.katago is not None and sync_katago:
            try:
                self.katago.play(player, None, None)
            except Exception:
                pass
        self._bot_timer = None

    def _human_pass(self):
        if self._is_human_turn():
            self._do_pass(sync_katago=True)

    def _undo(self):
        if not self.moves or self.mode != "play":
            return
        self._cancel_resign()
        drop = 2 if (self.opponent != OPP_HUMAN and len(self.moves) >= 2) else 1
        self.moves = self.moves[:-drop]
        self._rebuild()

    def _rebuild(self):
        self.board = GoBoard(self.n)
        for _, point in self.moves:
            if point is None:
                self.board.pass_move()
            else:
                self.board.play_move(*point)
        if self.katago is not None:
            try:
                self.katago.reset()
                for player, point in self.moves:
                    self.katago.play(player, *(point if point else (None, None)))
            except Exception:
                pass
        self._pending = None
        self.katago_thinking = False
        self._bot_timer = None

    def _resign(self):
        if self.mode != "play":
            return
        # Two-step: first press asks for confirmation so nobody resigns by accident.
        if not self._confirm_resign:
            self._confirm_resign = True
            self._status = "Resign? Press Resign / R again to confirm."
            return
        self._confirm_resign = False
        loser = self.board.current_player
        winner = "White" if loser == BLACK else "Black"
        self._winner_text = f"{winner} wins by resignation"
        self._score_text = ""
        self._finish_to_over()

    def _finish(self):
        b, w, winner = self.board.get_final_score()
        self._winner_text = "Tie game" if winner == "Tie" else f"{winner} wins"
        self._score_text = f"Black {b:.1f}   ·   White {w:.1f}"
        self._finish_to_over()

    def _finish_to_over(self):
        self.mode = "over"
        self._status = ""
        self._stop_katago()
        self._build_over_buttons()

    # ----------------------------------------------------------------- loop
    def update(self):
        if self.mode != "play":
            return
        if self.board.is_game_over():
            self._finish()
            return
        if self._is_human_turn():
            return
        # Bot's turn
        if self.opponent == OPP_RANDOM:
            now = pygame.time.get_ticks()
            if self.fast:
                self._random_move()
            elif self._bot_timer is None:
                self._bot_timer = now + 350
            elif now >= self._bot_timer:
                self._random_move()
        elif self.opponent == OPP_KATAGO:
            if self._pending is not None:
                r, c, raw = self._pending
                self._pending = None
                self.katago_thinking = False
                self._status = ""
                if raw == "resign" or raw.startswith("error:"):
                    self._winner_text = ("White" if self.board.current_player == BLACK else "Black") + " wins"
                    self._finish_to_over()
                elif raw == "pass" or r is None:
                    self._do_pass(sync_katago=False)
                else:
                    self._do_move(r, c, sync_katago=False)
            elif not self.katago_thinking:
                self.katago_thinking = True
                self._status = "KataGo is thinking…"
                threading.Thread(target=self._katago_think,
                                 args=(self.board.current_player,), daemon=True).start()

    def _random_move(self):
        legal = self.board.get_legal_moves()
        if legal and (len(self.moves) < 8 or random.random() < 0.85):
            self._do_move(*random.choice(legal), sync_katago=True)
        else:
            self._do_pass(sync_katago=True)

    def handle(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type == pygame.KEYDOWN:
            self._handle_key(event.key)
        for b in self.buttons:
            b.handle(event)
        if self.mode == "play":
            if event.type == pygame.MOUSEMOTION:
                self.hover = None
                if self._is_human_turn():
                    cell = self.view.cell_at(event.pos)
                    if cell and self.board.is_legal(*cell):
                        self.hover = cell
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._is_human_turn():
                    cell = self.view.cell_at(event.pos)
                    if cell and self.board.is_legal(*cell):
                        self._do_move(*cell, sync_katago=True)
                        self.hover = None

    def _handle_key(self, key):
        if self.mode == "menu":
            if key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self._begin_game()
            elif key == pygame.K_1:
                self._set_opponent(OPP_HUMAN)
            elif key == pygame.K_2:
                self._set_opponent(OPP_RANDOM)
            elif key == pygame.K_3:
                self._set_opponent(OPP_KATAGO)
            elif key == pygame.K_b:
                self._set_color(BLACK)
            elif key == pygame.K_w:
                self._set_color(WHITE)
            elif key in (pygame.K_q, pygame.K_ESCAPE):
                self.running = False
        elif self.mode == "play":
            if key == pygame.K_p:
                self._human_pass()
            elif key == pygame.K_u:
                self._undo()
            elif key == pygame.K_r:
                self._resign()
            elif key == pygame.K_s:
                self._toggle_mute()
            elif key in (pygame.K_m, pygame.K_ESCAPE):
                self._to_menu()
        elif self.mode == "over":
            if key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_n):
                self._rematch()
            elif key in (pygame.K_m, pygame.K_ESCAPE):
                self._to_menu()
            elif key == pygame.K_q:
                self.running = False

    def _toggle_mute(self):
        self.assets.muted = not self.assets.muted

    def _rematch(self):
        """Start a fresh game with the current opponent/colour settings."""
        self._begin_game()

    # --------------------------------------------------------------- buttons
    def _set_opponent(self, opp):
        self.opponent = opp
        self._build_menu()

    def _set_color(self, color):
        self.human_color = color
        self._build_menu()

    def _to_menu(self):
        self._stop_katago()
        self.mode = "menu"
        self._build_menu()

    def _build_menu(self):
        cx = theme.WINDOW_W // 2
        y = 300
        self.buttons = []
        for i, opp in enumerate(OPPONENTS):
            self.buttons.append(Button((cx - 240 + i * 165, y, 150, 48),
                                       f"vs {opp}", lambda o=opp: self._set_opponent(o),
                                       selected=(self.opponent == opp)))
        for i, opp in enumerate(OPPONENTS):
            self.buttons[i].hint = str(i + 1)
        y += 90
        self.buttons.append(Button((cx - 165, y, 150, 48), "Play Black",
                                   lambda: self._set_color(BLACK),
                                   selected=(self.human_color == BLACK), hint="B"))
        self.buttons.append(Button((cx + 15, y, 150, 48), "Play White",
                                   lambda: self._set_color(WHITE),
                                   selected=(self.human_color == WHITE), hint="W"))
        self.buttons.append(Button((cx - 90, y + 100, 180, 54), "Start Game",
                                   self._begin_game, kind="accent", hint="Enter"))

    def _build_play_buttons(self):
        bx = self.panel_x + 22
        bw = (theme.PANEL_W - 22 * 2 - 12) // 2
        by = theme.WINDOW_H - 134
        h = 46
        self.buttons = [
            Button((bx, by, bw, h), "Pass", self._human_pass, hint="P"),
            Button((bx + bw + 12, by, bw, h), "Undo", self._undo, hint="U"),
            Button((bx, by + h + 12, bw, h), "Resign", self._resign, kind="warn", hint="R"),
            Button((bx + bw + 12, by + h + 12, bw, h), "Menu", self._to_menu, hint="M"),
        ]

    def _build_over_buttons(self):
        cx = theme.WINDOW_W // 2
        y = 430
        self.buttons = [
            Button((cx - 285, y, 180, 52), "Rematch", self._rematch, kind="accent", hint="N"),
            Button((cx - 90, y, 180, 52), "Menu", self._to_menu, hint="M"),
            Button((cx + 105, y, 180, 52), "Quit", self._quit, kind="warn", hint="Q"),
        ]

    def _quit(self):
        self.running = False

    # ----------------------------------------------------------------- draw
    def draw(self):
        self.screen.fill(theme.BG)
        if self.mode == "menu":
            self._draw_menu()
        else:
            self.view.draw(self.screen, self.board, self.hover if self.mode == "play" else None,
                           self.board.current_player)
            self._draw_panel()
            if self.mode == "over":
                self._draw_over_overlay()
        for b in self.buttons:
            b.draw(self.screen)
        pygame.display.flip()

    def _draw_menu(self):
        cx = theme.WINDOW_W // 2
        title = theme.font(64, bold=True).render("ZetaGo", True, theme.TEXT)
        self.screen.blit(title, title.get_rect(center=(cx, 150)))
        sub = theme.font(22).render("7×7 Go · bitboard engine · KataGo", True, theme.TEXT_DIM)
        self.screen.blit(sub, sub.get_rect(center=(cx, 205)))
        for label, y in (("Opponent", 278), ("Your colour", 368)):
            t = theme.font(18, bold=True).render(label, True, theme.TEXT_DIM)
            self.screen.blit(t, t.get_rect(center=(cx, y)))
        hint = theme.font(15).render(
            "1/2/3 opponent   ·   B / W colour   ·   Enter to start",
            True, theme.TEXT_FAINT)
        self.screen.blit(hint, hint.get_rect(center=(cx, theme.WINDOW_H - 60)))

    def _coord(self, point):
        r, c = point
        return f"{_COLS[c]}{self.n - r}"

    def _last_move_text(self):
        if not self.moves:
            return "Last: —"
        player, point = self.moves[-1]
        who = "Black" if player == BLACK else "White"
        if point is None:
            return f"Last: {who} passed"
        return f"Last: {who} {self._coord(point)}"

    def _panel_text(self, text, x, y, size=20, color=theme.TEXT, bold=False):
        surf = theme.font(size, bold=bold).render(text, True, color)
        self.screen.blit(surf, (x, y))
        return y + surf.get_height()

    def _draw_panel(self):
        px = self.panel_x
        pygame.draw.rect(self.screen, theme.PANEL_BG, (px, 0, theme.PANEL_W, theme.WINDOW_H))
        pygame.draw.line(self.screen, theme.PANEL_LINE, (px, 0), (px, theme.WINDOW_H), 2)
        x = px + 22
        y = 24
        y = self._panel_text("ZetaGo", x, y, 34, theme.TEXT, bold=True) + 6
        y = self._panel_text(f"Opponent: {self.opponent}", x, y, 18, theme.TEXT_DIM) + 2
        you = "Black" if self.human_color == BLACK else "White"
        if self.opponent != OPP_HUMAN:
            y = self._panel_text(f"You: {you}", x, y, 18, theme.TEXT_DIM) + 10
        else:
            y += 12

        # Turn indicator
        turn = "Black" if self.board.current_player == BLACK else "White"
        pygame.draw.circle(self.screen, theme.BLACK_DOT if turn == "Black" else theme.WHITE_DOT,
                           (x + 10, y + 14), 10)
        pygame.draw.circle(self.screen, theme.PANEL_LINE, (x + 10, y + 14), 10, 1)
        self._panel_text(f"{turn} to move", x + 30, y + 3, 22, theme.TEXT, bold=True)
        y += 44

        y = self._panel_text(f"Move:     {self.board.move_number}", x, y, 19, theme.TEXT_DIM) + 2
        y = self._panel_text(f"Captures: B {self.board.captured_by_black}  ·  "
                             f"W {self.board.captured_by_white}", x, y, 19, theme.TEXT_DIM) + 2
        y = self._panel_text(self._last_move_text(), x, y, 19, theme.TEXT_DIM) + 12

        # Live area-score estimate
        b, w, leader = self.board.get_final_score()
        y = self._panel_text("Score (area est.)", x, y, 17, theme.TEXT_DIM, bold=True) + 2
        y = self._panel_text(f"Black {b:.1f}   ·   White {w:.1f}", x, y, 19, theme.TEXT) + 2
        lead_col = theme.GOOD if leader != "Tie" else theme.TEXT_DIM
        y = self._panel_text(f"Leader: {leader}", x, y, 18, lead_col) + 12

        # KataGo thinking indicator (animated) or a status/prompt line.
        if self.katago_thinking:
            dots = "." * (1 + (pygame.time.get_ticks() // 350) % 3)
            y = self._panel_text(f"KataGo is thinking{dots}", x, y, 17, theme.ACCENT) + 2
        elif self._status:
            col = theme.WARN if self._confirm_resign else theme.ACCENT
            for line in self._wrap(self._status, 30):
                y = self._panel_text(line, x, y, 16, col) + 2

        # Controls hint + sound state, pinned near the bottom above the buttons.
        hy = theme.WINDOW_H - 176
        sound = "on" if not self.assets.muted else "off"
        self._panel_text("P Pass · U Undo · R Resign", x, hy, 14, theme.TEXT_FAINT)
        self._panel_text(f"M Menu · S Sound ({sound})", x, hy + 18, 14, theme.TEXT_FAINT)

    def _draw_over_overlay(self):
        overlay = pygame.Surface((self.panel_x, theme.WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        cx = theme.WINDOW_W // 2
        t = theme.font(40, bold=True).render("Game Over", True, theme.TEXT)
        self.screen.blit(t, t.get_rect(center=(cx, 300)))
        r = theme.font(26, bold=True).render(self._winner_text, True, theme.GOOD)
        self.screen.blit(r, r.get_rect(center=(cx, 352)))
        if self._score_text:
            s = theme.font(22).render(self._score_text, True, theme.TEXT)
            self.screen.blit(s, s.get_rect(center=(cx, 390)))

    @staticmethod
    def _wrap(text, width):
        words, lines, cur = text.split(), [], ""
        for wd in words:
            if len(cur) + len(wd) + 1 > width:
                lines.append(cur)
                cur = wd
            else:
                cur = (cur + " " + wd).strip()
        if cur:
            lines.append(cur)
        return lines

    # ------------------------------------------------------------------ run
    def run(self):
        self.running = True
        while self.running:
            for event in pygame.event.get():
                self.handle(event)
            self.update()
            self.draw()
            self.clock.tick(theme.FPS)
        self._stop_katago()
        pygame.quit()

    def selftest(self, max_steps=600):
        """Headless smoke test: render menu, auto-play a full Random game, render game-over."""
        self.draw()                       # menu renders
        self.opponent = OPP_RANDOM
        self.human_color = BLACK
        self.fast = True
        self._begin_game()
        steps = 0
        while self.mode != "over" and steps < max_steps:
            if self.mode == "play" and self._is_human_turn():
                legal = self.board.get_legal_moves()
                if legal and random.random() < 0.9:
                    self._do_move(*random.choice(legal), sync_katago=False)
                else:
                    self._do_pass(sync_katago=False)
            self.update()
            self.draw()
            steps += 1
        self.draw()                       # game-over overlay renders
        pygame.quit()
        return self.mode, self.board.move_number
