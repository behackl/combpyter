import pytest

from combpyter import DyckPath, DyckPaths


def test_dyck_path_number():
    counts = [len(DyckPaths(nn)) for nn in range(1, 6)]
    assert counts == [1, 2, 5, 14, 42]

def test_dyck_path_check_unbalanced():
    with pytest.raises(ValueError):
        DyckPath([1, 1, 0])

def test_dyck_path_check_excursion():
    with pytest.raises(ValueError):
        DyckPath([1, 0, 0, 1])

def test_simple_path():
    path = DyckPath([1, 0, 1, 0, 1, 1, 0, 1, 0, 0])
    assert repr(path) == "Dyck path with steps [1, 0, 1, 0, 1, 1, 0, 1, 0, 0]"
    assert path.peaks() == [0, 2, 5, 7]
    assert path.valleys() == [1, 3, 6]

def test_generation():
    paths = set(DyckPaths(3))
    assert paths == set([
        DyckPath([1, 1, 1, 0, 0, 0]),
        DyckPath([1, 0, 1, 1, 0, 0]),
        DyckPath([1, 0, 1, 0, 1, 0]),
        DyckPath([1, 1, 0, 1, 0, 0]),
        DyckPath([1, 1, 0, 0, 1, 0]),
    ])

def test_random_path():
    random_path = DyckPaths(42).random_element()
    assert random_path.semilength == 42

def test_height_profile():
    path = DyckPath([1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0])
    assert path.height_profile() == [0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 2, 1, 0, 1, 0]


def test_last_upstep_expansion():
    path = DyckPath([1, 1, 0, 1, 0, 0])
    assert set(path.last_upstep_expansion()) == set([
        DyckPath([1, 1, 0, 1, 1, 0, 0, 0]),
        DyckPath([1, 1, 0, 1, 0, 1, 0, 0]),
        DyckPath([1, 1, 0, 1, 0, 0, 1, 0]),
    ])

def test_last_upstep_reduction():
    path = DyckPath([1, 1, 0, 1, 0, 0])
    assert path.last_upstep_reduction() == DyckPath([1, 1, 0, 0])
