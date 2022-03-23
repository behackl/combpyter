from __future__ import annotations

import numpy as np
from math import comb

__all__ = [
        "DyckPath",
        "DyckPaths",
]


class DyckPath:
    """Sequences of up- and down-steps that represent Dyck paths."""
    def __init__(self, steps, check=True) -> None:
        if isinstance(steps, DyckPath):
            steps = steps.steps
        if check:
            n = len(steps) // 2
            assert sum(steps) == n, "Up- and Down-steps are not balanced."
            assert all([sum(steps[:k]) >= k/2 for k in range(n)]), "Path does not describe an excursion"
        self.steps = steps
    
    def __len__(self):
        return len(self.steps)
    
    def __repr__(self):
        return f"Dyck path with steps {self.steps}"

    def __iter__(self):
        return iter(self.steps)

    def __hash__(self):
        return hash(tuple(self.steps))

    def __eq__(self, other):
        return self.steps == other.steps

    @property
    def semilength(self):
        return len(self.steps) // 2

    def pretty_repr(self):
        string_mat = [[" " if i <= j else "." for j in range(self.semilength+1)]
            for i in range(self.semilength+1)]
        string_mat[0][0] = "_"
        i, j = 0, 0
        for step in self.steps:
            if step == 1:
                j += 1
                string_mat[i][j] = "_"
            else:
                i += 1
                string_mat[i][j] = "|"
        for i in range(1, self.semilength+1):
            for j in range(i, self.semilength + 1):
                if string_mat[i][j] == " ":
                    string_mat[i][j] = "o"
                elif string_mat[i][j] in ("|", "_"):
                    break
        return "\n".join(["".join(row) for row in string_mat])

    def height_profile(self):
        """Returns the height profile of the path."""
        return [0] + list(np.cumsum(2 * np.array(self.steps) - 1))
    
    def height(self):
        """Returns the height of the path."""
        return max(self.height_profile())

    def peaks(self):
        """Starting indices of the peaks of this path.

        A peak is an occurrence of the pattern 1, 0 (an up-step
        followed by a down-step).
        
        Examples
        --------

        >>> path = DyckPath([1, 1, 0, 1, 0, 1, 0, 0])
        >>> path.peaks()
        [1, 3, 5]
        """
        return [ind for ind in range(len(self) - 1)
                if self.steps[ind] == 1 and self.steps[ind+1] == 0]

    def valleys(self):
        """Starting indices of the valleys of this path.
        
        Examples
        --------

        >>> path = DyckPath([1, 1, 0, 1, 0, 1, 0, 0])
        >>> path.valleys()
        [2, 4]
        """
        return [ind for ind in range(len(self) - 1)
                if self.steps[ind] == 0 and self.steps[ind+1] == 1]
    
    def last_upstep_expansion(self):
        """Generator for Dyck paths that can be obtained by expanding this path.
        
        By expanding a path, a new peak is inserted in all possible
        positions along the last ascent.
        """
        PathClass = type(self)
        last_peak = self.peaks()[-1]
        for pos in range(last_peak+1, len(self)+1):
            prefix, postfix = self.steps[:pos], self.steps[pos:]
            yield PathClass(prefix + [1, 0] + postfix, check=False)

    def last_upstep_reduction(self):
        """The Dyck path resulting from removing the last peak."""
        last_peak = self.peaks()[-1]
        return type(self)(self.steps[:last_peak] + self.steps[last_peak+2:], check=False)

    def plot(self, **kwargs):
        """Plots the Dyck path as a line diagram."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(**kwargs)
        ax.plot(self.height_profile())
        ax.set_xticks(range(len(self) + 1))
        ax.set_yticks(range(self.semilength + 1))
        ax.grid()
        ax.set_aspect(1)
        return ax

    def ascent_descent_code(self) -> tuple(list, list):
        """The ascent-descent code of a Dyck path.

        The ascent-descent code is a tuple ``(A, D)``, where
        ``A`` and ``D`` are lists containing the ascent and
        descent lengths of the path, respectively.

        Examples
        --------

        >>> pth = DyckPath([1, 1, 0, 1, 1, 0, 0, 0])
        >>> pth.ascent_descent_code()
        ([2, 2], [1, 3])
        """
        descent_ascent_code = ([], [0])
        current_type = 1
        for step in self.steps:
            if step == current_type:
                descent_ascent_code[current_type][-1] += 1
            else:
                current_type = 1 - current_type
                descent_ascent_code[current_type].append(1)

        descents, ascents = descent_ascent_code
        return ascents, descents

    @classmethod
    def from_ascent_descent_code(cls, ascents: list, descents: list) -> cls:
        """The Dyck path corresponding to the specified ascent-descent code.

        Examples
        --------

        >>> ascents, descents = [2, 2], [1, 3]
        >>> DyckPath.from_ascent_descent_code(ascents, descents)
        Dyck path with steps [1, 1, 0, 1, 1, 0, 0, 0]
        """
        if semilength := len(ascents) != len(descents):
            raise ValueError("The specified ascent and descent lists do not have the same length")
        step_sequence = []
        for asc, desc in zip(ascents, descents):
            step_sequence.extend([1] * asc + [0] * desc)

        return cls(step_sequence)

    def reverse_involution(self):
        """The path resulting from taking the complement of the reversed path.

        References
        ----------

        `FindStat Map 00028 <https://www.findstat.org/MapsDatabase/Mp00028/>`__

        Examples
        --------

        >>> path = DyckPath([1, 1, 0, 0, 1, 0])
        >>> path.reverse_involution()
        Dyck path with steps [1, 0, 1, 1, 0, 0]
        """
        return type(self)([1 - step for step in reversed(self.steps)])


    def lalanne_kreweras_involution(self):
        """The path resulting from applying the Lalanne-Kreweras involution.

        The involution is constructed by labeling up- and down-steps and
        denoting all starting indices of double rises and double falls,
        respectively. After appending the semilength of the path to both
        lists, the forwards differences describe the ascent-descent code of
        the corresponding Dyck path.

        References
        ----------

        `FindStat Map 00120 <https://www.findstat.org/MapsDatabase/Mp00120>`__

        Examples
        --------
        
        >>> path = DyckPath([1, 1, 1, 0, 0, 0])
        >>> path.lalanne_kreweras_involution()
        Dyck path with steps [1, 0, 1, 0, 1, 0]
        """
        double_rises = []
        double_falls = []
        current_upstep = 0
        current_downstep = 0
        for ind in range(len(self) - 1):
            if self.steps[ind] == 1:
                current_upstep += 1
                if self.steps[ind + 1] == 1:
                    double_rises.append(current_upstep)
            else:
                current_downstep += 1
                if self.steps[ind + 1] == 0:
                    double_falls.append(current_downstep)
        
        ascents = np.diff([0] + double_rises + [self.semilength])
        descents = np.diff([0] + double_falls + [self.semilength])
        return type(self).from_ascent_descent_code(ascents, descents)


class DyckPaths:
    """Generator object for all Dyck paths of given semi-length."""
    element_class = DyckPath

    def __init__(self, semilength):
        self.semilength = semilength
    
    def __len__(self):
        n = self.semilength
        return comb(2*n, n) // (n+1)

    def __iter__(self):
        if self.semilength == 0:
            yield self.element_class([])
        elif self.semilength == 1:
            yield self.element_class([1, 0])
        else:
            for path in type(self)(self.semilength - 1):
                for expanded_path in path.last_upstep_expansion():
                    yield expanded_path
    
    def random_element(self):
        """Generate a random Dyck path in this family.

        Uses a random permutation of n down-steps and (n+1)
        up-steps + constructs the correct cyclic shift
        to turn the word starting at index 1 to a Dyck path.
        """
        steps = [0] * self.semilength + [1] * (self.semilength + 1)
        steps = np.random.permutation(steps)
        sloped_steps = (2*self.semilength + 1) * steps - (self.semilength + 1)
        ind = np.argmin(np.cumsum(sloped_steps)) + 1
        dyck_steps = list(steps[ind:]) + list(steps[:ind])
        return self.element_class(dyck_steps[1:])
