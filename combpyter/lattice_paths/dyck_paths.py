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
        """Returns all starting indices of peaks.
        
        Examples
        --------

        >>> path = DyckPath([1, 1, 0, 1, 0, 1, 0, 0])
        >>> path.peaks()
        [1, 3, 5]
        """
        return [ind for ind in range(len(self) - 1)
                if self.steps[ind] == 1 and self.steps[ind+1] == 0]

    def valleys(self):
        """Returns all starting indices of valleys.
        
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
        last_peak = self.peaks()[-1]
        for pos in range(last_peak+1, len(self)+1):
            prefix, postfix = self.steps[:pos], self.steps[pos:]
            yield DyckPath(prefix + [1, 0] + postfix, check=False)

    def last_upstep_reduction(self):
        """The path resulting from removing the last peak."""
        last_peak = self.peaks()[-1]
        return DyckPath(self.steps[:last_peak] + self.steps[last_peak+2:], check=False)

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



class DyckPaths:
    """Generator object for all Dyck paths of given semi-length."""
    def __init__(self, semilength):
        self.semilength = semilength
    
    def __len__(self):
        n = self.semilength
        return comb(2*n, n) // (n+1)

    def __iter__(self):
        if self.semilength == 0:
            yield DyckPath([])
        elif self.semilength == 1:
            yield DyckPath([1, 0])
        else:
            for path in DyckPaths(self.semilength - 1):
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
        return DyckPath(dyck_steps[1:])
