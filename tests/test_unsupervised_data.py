"""Track B data path: unlabelled discipline and dedup/holdout mechanics (task 3.0a)."""

import numpy as np
import pytest

from training.unsupervised.data import _assert_unlabelled, load_unlabelled_train


def test_assert_unlabelled_raises_on_leaked_label():
    with pytest.raises(AssertionError):
        _assert_unlabelled({"states": None, "moves": None})


def test_assert_unlabelled_passes_states_only():
    _assert_unlabelled({"states": np.zeros(1)})  # must not raise


def _write_synthetic_h5(path, n_games=4, moves_per_game=5, n_dupes=3):
    import h5py

    rng = np.random.default_rng(0)
    states, moves, values, margins, players, game_id, move_no = [], [], [], [], [], [], []
    for g in range(n_games):
        for m in range(moves_per_game):
            st = np.zeros((6, 7, 7), dtype=np.uint8)
            st[0, m % 7, g % 7] = 1  # varies with (game, move) -> distinct positions
            states.append(st)
            moves.append(m % 49)
            values.append(1 if m % 2 == 0 else -1)
            margins.append(float(m))
            players.append(1 if m % 2 == 0 else -1)
            game_id.append(g)
            move_no.append(m)
    # duplicate the very first position across a few extra rows/games, so
    # apply_dedup(..., "unique") has something real to collapse.
    for k in range(n_dupes):
        states.append(states[0])
        moves.append(moves[0])
        values.append(values[0])
        margins.append(margins[0])
        players.append(players[0])
        game_id.append(1000 + k)
        move_no.append(0)

    with h5py.File(path, "w") as h:
        h.create_dataset("states", data=np.stack(states))
        h.create_dataset("moves", data=np.array(moves, dtype=np.int16))
        h.create_dataset("values", data=np.array(values, dtype=np.int8))
        h.create_dataset("margins", data=np.array(margins, dtype=np.float32))
        h.create_dataset("players", data=np.array(players, dtype=np.int8))
        h.create_dataset("game_id", data=np.array(game_id, dtype=np.uint32))
        h.create_dataset("move_no", data=np.array(move_no, dtype=np.int32))

    n_unique = n_games * moves_per_game  # the n_dupes rows collapse into row 0
    return n_unique


def test_load_unlabelled_train_shapes_and_dedup(tmp_path):
    h5_path = str(tmp_path / "synthetic_train.h5")
    n_unique = _write_synthetic_h5(h5_path)

    fit, holdout = load_unlabelled_train(h5_path, holdout_frac=0.2, seed=0)

    assert fit.dtype == np.float32
    assert holdout.dtype == np.float32
    assert fit.shape[1:] == (2, 7, 7)
    assert holdout.shape[1:] == (2, 7, 7)
    assert len(fit) + len(holdout) == n_unique  # dedup collapsed the duplicated rows


def test_load_unlabelled_train_is_reproducible(tmp_path):
    h5_path = str(tmp_path / "synthetic_train.h5")
    _write_synthetic_h5(h5_path)

    fit1, hold1 = load_unlabelled_train(h5_path, holdout_frac=0.2, seed=7)
    fit2, hold2 = load_unlabelled_train(h5_path, holdout_frac=0.2, seed=7)

    np.testing.assert_array_equal(fit1, fit2)
    np.testing.assert_array_equal(hold1, hold2)
