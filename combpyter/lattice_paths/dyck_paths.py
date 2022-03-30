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
            if sum(steps) != n:
                raise ValueError(
                        "Up- and down-steps in the step sequence" 
                        f" {steps} are not balanced"
                    )
            cumulated_steps = [0] + list(np.cumsum(steps))
            if not all([cumulated_steps[k] >= k/2 for k in range(len(steps) + 1)]):
                raise ValueError(
                        "The path described by the step sequence"
                        f" {steps} does not describe an excursion"
                    )
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

    def peaks(self, order: int | None = None) -> list(int):
        """Starting indices of the peaks of this path.

        A peak is an occurrence of the pattern 1, 0 (an up-step
        followed by a down-step).

        Parameters
        ----------

        order
            If set to an integer d, only the starting indices
            of d-peaks are returned.
            
        
        Examples
        --------

        >>> path = DyckPath([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])
        >>> path.peaks()
        [1, 5, 7]
        >>> path.peaks(order=1)
        [5, 7]
        >>> path.peaks(order=2)
        [0]
        >>> path.peaks(order=3)
        []
        """
        if order is None:
            return [ind for ind in range(len(self) - 1)
                    if self.steps[ind] == 1 and self.steps[ind+1] == 0]

        ascents, descents = self.ascent_descent_code()
        peak_index_orders = []
        current_index = 0
        for a, d in zip(ascents, descents):
            peak_order = min(a, d)
            current_index += a
            peak_index_orders.append((current_index - peak_order, peak_order))
            current_index += d
        return [ind for (ind, ord) in peak_index_orders if ord == order]

    def valleys(self, order: int | None = None) -> list(int):
        """Starting indices of the valleys of this path.

        A valley is an occurrence of the pattern 0, 1 (a down-step
        followed by an up-step).

        Parameters
        ----------

        order
            If set to an integer d, only the starting indices
            of d-valleys are returned.
        
        Examples
        --------

        >>> path = DyckPath([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])
        >>> path.valleys()
        [3, 6]
        >>> path.valleys(order=1)
        [6]
        >>> path.valleys(order=2)
        [2]
        >>> path.valleys(order=3)
        []
        """
        if order is None:
            return [ind for ind in range(len(self) - 1)
                    if self.steps[ind] == 0 and self.steps[ind+1] == 1]

        ascents, descents = self.ascent_descent_code()
        valley_index_orders = []
        current_index = ascents[0]
        for ind in range(len(ascents) - 1):
            valley_order = min(descents[ind], ascents[ind+1])
            current_index += descents[ind]
            valley_index_orders.append((current_index - valley_order, valley_order))
            current_index += ascents[ind + 1]

        return [ind for (ind, ord) in valley_index_orders if ord == order]

    def falls(self, order: int | None = None) -> list(int):
        """Starting indices of falls of this path.

        Parameters
        ----------

        order
            If order is set to a positive integer d, only the
            starting indices of d-falls (maximal sequences of d
            consecutive down-steps) are returned.


        Examples
        --------

        >>> path = DyckPath([1, 1, 0, 1, 0, 1, 0, 0])
        >>> path.falls()
        [2, 4, 6]
        >>> path.falls(order=2)
        [6]
        >>> path.falls(order=1)
        [2, 4]
        """
        ascents, descents = self.ascent_descent_code()
        ascent_index_order = []
        current_index = 0
        for a, d in zip(ascents, descents): 
            current_index += a
            ascent_index_order.append((current_index, d))
            current_index += d

        if order is not None:
            ascent_index_order = [
                (ind, ord) for (ind, ord) in ascent_index_order 
                if ord == order
            ]
        return [ind for (ind, ord) in ascent_index_order]


    def rises(self, order: int | None = None) -> list(int):
        """Starting indices of rises of this path.

        Parameters
        ----------

        order
            If order is set to a positive integer d, only the
            starting indices of d-rises (maximal sequences of d
            consecutive up-steps) are returned.


        Examples
        --------

        >>> path = DyckPath([1, 1, 0, 1, 0, 1, 0, 0])
        >>> path.rises()
        [0, 3, 5]
        >>> path.rises(order=2)
        [0]
        >>> path.rises(order=1)
        [3, 5]
        """
        ascents, descents = self.ascent_descent_code()
        ascent_index_order = []
        current_index = 0
        for a, d in zip(ascents, descents): 
            ascent_index_order.append((current_index, a))
            current_index += a + d

        if order is not None:
            ascent_index_order = [
                (ind, ord) for (ind, ord) in ascent_index_order 
                if ord == order
            ]
        return [ind for (ind, ord) in ascent_index_order]

    
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

    def plot(self, axes=None, **kwargs):
        """Plots the Dyck path as a line diagram.

        Parameters
        ----------

        axes
            The matplotlib axes that should be used to plot the path.
            If None, a new one will be created.
        """
        if axes is None:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(**kwargs)
            
        axes.plot(self.height_profile())
        axes.set_xticks(range(len(self) + 1))
        axes.set_yticks(range(self.semilength + 1))
        axes.grid()
        axes.set_aspect(1)
        return axes

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
        
        >>> path = DyckPath([1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0])
        >>> path.lalanne_kreweras_involution()
        Dyck path with steps [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0]
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
        
        ascents = np.diff([0] + double_falls + [self.semilength])
        descents = np.diff([0] + double_rises + [self.semilength])
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
