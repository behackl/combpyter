# combpyter

A lightweight Python library for generating and analyzing combinatorial objects.


## Installation

The latest release is simply pip-installable via PyPI. Simply run
```sh
pip install combpyter
```
to install the library.


## Features

Please note that this is a personal and mostly research-driven project.
As such, the implemented combinatorial structures span, for the time being,
a narrow scope.

Currently, `combpyter` supports the following combinatorial objects:

- Dyck paths (elements: `DyckPath`, generator: `DyckPaths`)


## Example usage

This snippet iterates over all Dyck paths of semi-length 8 and computes
the distribution of the number of peaks (which is a statistic described
by [Narayana numbers](https://en.wikipedia.org/wiki/Narayana_number)).

```python
>>> from combpyter import DyckPaths
>>> from collections import defaultdict
>>> peak_distribution = defaultdict(int)
>>> for path in DyckPaths(8):
...     num_peaks = len(path.peaks())
...     peak_distribution[num_peaks] += 1
...
>>> for num_peaks, num_paths in peak_distribution.items():
...     print(f"There are {num_paths} Dyck paths with {num_peaks} peaks.")
...
There are 1 Dyck paths with 1 peaks.
There are 28 Dyck paths with 2 peaks.
There are 196 Dyck paths with 3 peaks.
There are 490 Dyck paths with 4 peaks.
There are 490 Dyck paths with 5 peaks.
There are 196 Dyck paths with 6 peaks.
There are 28 Dyck paths with 7 peaks.
There are 1 Dyck paths with 8 peaks.
```
