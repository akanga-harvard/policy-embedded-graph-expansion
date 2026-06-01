# Standard library imports

# Third-party imports
import numpy as np

# Local imports
from core.linear_function import LinearFunction

class PiecewiseLinearFunction:
    """
    0 < changepoints[0] < changepoints[1] < ... < changepoints[-1] = 1/(1-beta)
    f(0, changepoints[0]) = value_functions[0]
    f(changepoints[0], changepoints[1]) = value_functions[1]
    ...
    f(changepoints[-2], changepoints[-1]) = value_functions[-1]
    """
    def __init__(self, changepoints: list[float], value_functions: list[LinearFunction]) -> None:
        assert len(changepoints) >= 1
        assert len(changepoints) == len(value_functions)
        assert changepoints[0] > 0
        assert all(changepoints[i] < changepoints[i+1] for i in range(len(changepoints)- 1))
        self.changepoints = changepoints
        self.value_functions = value_functions

    # Remove empty pieces and merge consecutie intervals with same function
    def clean(self):
        # print("BEFORE")
        # print(self)
        cleaned_changepoints = []
        cleaned_value_functions = []
        current_piece_b = 0
        current_linear_function = None
        for i in range(len(self.value_functions)):
            a = self.changepoints[i-1] if i > 0 else 0
            b = self.changepoints[i]
            if a == b:
                # Ignore zero length piece
                continue
            else:
                if current_linear_function is None:
                    current_piece_b = b
                    current_linear_function = self.value_functions[i]
                elif (current_linear_function.a == self.value_functions[i].a
                      and current_linear_function.b == self.value_functions[i].b):
                    # Extend interval of current piece
                    current_piece_b = b
                else:
                    # Close previous piece
                    cleaned_changepoints.append(current_piece_b)
                    cleaned_value_functions.append(current_linear_function)

                    # Start new piece
                    current_piece_b = b
                    current_linear_function = self.value_functions[i]
        # Close last piece
        cleaned_changepoints.append(current_piece_b)
        cleaned_value_functions.append(current_linear_function)
        self.changepoints = cleaned_changepoints
        self.value_functions = cleaned_value_functions
        # print("AFTER")
        # print(self)

    def __repr__(self):
        parts = []
        for i in range(len(self.value_functions)):
            a = self.changepoints[i-1] if i > 0 else 0
            b = self.changepoints[i]
            val_str = repr(self.value_functions[i])
            parts.append(f"[{a:.4f}, {b:.4f}): {val_str}")
        return "PiecewiseLinearFunction(\n  " + ",\n  ".join(parts) + "\n)"

    def _validate(self, startpoint_val: float, endpoint_val: float):
        assert np.isclose(self.value_functions[0](0), startpoint_val)
        assert np.isclose(self.value_functions[-1](self.changepoints[-1]), endpoint_val)

    def _merge_changepoints_with_indices(self, cp1: list[float], cp2: list[float], tol: float = 1e-12):
        merged, index1, index2 = [], [], []
        i = j = seg1 = seg2 = 0
        while i < len(cp1) or j < len(cp2):
            c1 = cp1[i] if i < len(cp1) else float("inf")
            c2 = cp2[j] if j < len(cp2) else float("inf")

            # Decide which changepoint to take next (treat equal within tol as the same point)
            if abs(c1 - c2) < tol:
                cp = c1
                i += 1
                j += 1
            elif c1 < c2:
                cp = c1
                i += 1
            else:
                cp = c2
                j += 1

            # Only add if it's not a duplicate (within tol) of the one we just recorded
            if not merged or abs(cp - merged[-1]) > tol:
                merged.append(cp)
                index1.append(seg1)
                index2.append(seg2)

            # Crossing a changepoint moves you to the *next* segment of whichever function(s) owned that changepoint
            if abs(cp - c1) < tol:
                seg1 += 1
            if abs(cp - c2) < tol:
                seg2 += 1

        return merged, index1, index2

    def __add__(self, other: "PiecewiseLinearFunction") -> "PiecewiseLinearFunction":
        new_changepoints, self_map, other_map = self._merge_changepoints_with_indices(self.changepoints, other.changepoints)
        value_functions = [
            LinearFunction(
                self.value_functions[self_map[i]].a + other.value_functions[other_map[i]].a,
                self.value_functions[self_map[i]].b + other.value_functions[other_map[i]].b
            )
            for i in range(len(new_changepoints))
        ]
        output = PiecewiseLinearFunction(new_changepoints, value_functions)

        # Check if addition is done properly
        for i in range(len(new_changepoints)):
            left = new_changepoints[i-1] if i > 0 else 0
            right = new_changepoints[i]
            mid = (left + right) / 2
            self_func = self.value_functions[self_map[i]]
            other_func = other.value_functions[other_map[i]]
            output_func = output.value_functions[i]
            assert np.isclose(self_func(left) + other_func(left), output_func(left))
            assert np.isclose(self_func(right) + other_func(right), output_func(right))
            assert np.isclose(self_func(mid) + other_func(mid), output_func(mid))

        return output

    def __mul__(self, other: "PiecewiseLinearFunction") -> "PiecewiseLinearFunction":
        # This should only be called with both functions are piecewise constant
        for i in range(len(self.value_functions)):
            assert self.value_functions[i].a == 0
        for i in range(len(other.value_functions)):
            assert other.value_functions[i].a == 0

        new_changepoints, self_map, other_map = self._merge_changepoints_with_indices(self.changepoints, other.changepoints)
        value_functions = [
            LinearFunction(
                0.0,
                self.value_functions[self_map[i]].b * other.value_functions[other_map[i]].b
            )
            for i in range(len(new_changepoints))
        ]
        output = PiecewiseLinearFunction(new_changepoints, value_functions)

        # Check if multiplication is done properly
        for i in range(len(new_changepoints)):
            left = new_changepoints[i-1] if i > 0 else 0
            right = new_changepoints[i]
            mid = (left + right) / 2
            self_func = self.value_functions[self_map[i]]
            other_func = other.value_functions[other_map[i]]
            output_func = output.value_functions[i]
            assert np.isclose(self_func(left) * other_func(left), output_func(left))
            assert np.isclose(self_func(right) * other_func(right), output_func(right))
            assert np.isclose(self_func(mid) * other_func(mid), output_func(mid))

        return output
    
    def mult_by_const(self, c: float) -> "PiecewiseLinearFunction":
        new_value_functions = [
            LinearFunction(c * self.value_functions[i].a, c * self.value_functions[i].b)
            for i in range(len(self.value_functions))
        ]
        output = PiecewiseLinearFunction(self.changepoints.copy(), new_value_functions)

        # Check if multiplication is done properly
        for i in range(len(self.changepoints)):
            left = self.changepoints[i-1] if i > 0 else 0
            right = self.changepoints[i]
            mid = (left + right) / 2
            self_func = self.value_functions[i]
            output_func = output.value_functions[i]
            assert np.isclose(self_func(left) * c, output_func(left))
            assert np.isclose(self_func(right) * c, output_func(right))
            assert np.isclose(self_func(mid) * c, output_func(mid))

        return output
    
    def add_const(self, c: float) -> "PiecewiseLinearFunction":
        new_value_functions = [
            LinearFunction(self.value_functions[i].a, c + self.value_functions[i].b)
            for i in range(len(self.value_functions))
        ]
        output = PiecewiseLinearFunction(self.changepoints.copy(), new_value_functions)

        # Check if addition is done properly
        for i in range(len(self.changepoints)):
            left = self.changepoints[i-1] if i > 0 else 0
            right = self.changepoints[i]
            mid = (left + right) / 2
            self_func = self.value_functions[i]
            output_func = output.value_functions[i]
            assert np.isclose(self_func(left) + c, output_func(left))
            assert np.isclose(self_func(right) + c, output_func(right))
            assert np.isclose(self_func(mid) + c, output_func(mid))

        return output

    def derivative(self) -> "PiecewiseLinearFunction":
        derivs = [val_fn.derivative() for val_fn in self.value_functions]
        return PiecewiseLinearFunction(self.changepoints.copy(), derivs)
    
    def integrate_piecewise_constant(self):
        assert all(func.a == 0 for func in self.value_functions)
        value_functions = []
        total_area = 0.0
        prev_cp = 0.0
        for i in range(len(self.changepoints)):
            slope = self.value_functions[i].b
            intercept = total_area - slope * prev_cp
            value_functions.append(LinearFunction(slope, intercept))
            total_area += slope * (self.changepoints[i] - prev_cp)
            prev_cp = self.changepoints[i]
        output = PiecewiseLinearFunction(self.changepoints.copy(), value_functions)

        # Check if integration function is constructed properly
        for i in range(len(self.changepoints)):
            x_prev = self.changepoints[i-1] if i > 0 else 0
            x_now = self.changepoints[i]
            x_mid = (x_now + x_prev) / 2
            func = output.value_functions[i]
            assert np.isclose(func(x_prev), self.integrate_zero_to_x(x_prev), atol=1e-6)
            assert np.isclose(func(x_now), self.integrate_zero_to_x(x_now), atol=1e-6)
            assert np.isclose(func(x_mid), self.integrate_zero_to_x(x_mid), atol=1e-6)

        return output

    def integrate_zero_to_x(self, x: float) -> float:
        assert 0 <= x and x <= self.changepoints[-1]
        val = 0
        for i in range(len(self.changepoints)):
            a = self.changepoints[i-1] if i > 0 else 0
            b = self.changepoints[i]
            if x < a:
                break
            else:
                left = a
                right = min(x,b)
                left_val = self.value_functions[i](left)
                right_val = self.value_functions[i](right)
                val += np.trapezoid([left_val, right_val], [left, right])
        return val
    
    def max_with_linear(self) -> "PiecewiseLinearFunction":
        changepoints_map = []
        new_changepoints = []
        new_value_functions = []
        for i in range(len(self.changepoints)):
            left = self.changepoints[i-1] if i > 0 else 0
            right = self.changepoints[i]
            
            # Compare f with linear function m
            # This segment either remains the same f, replaced by m, a new changepoint is created in-between
            func = self.value_functions[i]
            assert func.a >= 0
            if func.a != 1:
                crossing_point = func.b / (1 - func.a)
                if crossing_point <= left:
                    # Entire piece is just x
                    changepoints_map.append(i)
                    new_changepoints.append(right)
                    new_value_functions.append(LinearFunction(1.0, 0.0))
                elif crossing_point < right:
                    # First piece is func, then second piece is x
                    changepoints_map += [i,i]
                    new_changepoints += [crossing_point, right]
                    new_value_functions.append(func)
                    new_value_functions.append(LinearFunction(1.0, 0.0))
                else:
                    # Crosses after right, entire piece is just func
                    changepoints_map.append(i)
                    new_changepoints.append(right)
                    new_value_functions.append(func)
            else:
                changepoints_map.append(i)
                new_changepoints.append(right)
                if func.b >= 0:
                    # Entire piece is just func
                    new_value_functions.append(func)
                else:
                    # Entire piece is just x
                    new_value_functions.append(LinearFunction(1.0, 0.0))
        output = PiecewiseLinearFunction(new_changepoints, new_value_functions)

        # Check if max is done properly
        for i in range(len(new_changepoints)):
            left = new_changepoints[i-1] if i > 0 else 0
            right = new_changepoints[i]
            mid = (left + right) / 2
            self_func = self.value_functions[changepoints_map[i]]
            output_func = output.value_functions[i]
            assert np.isclose(max(left, self_func(left)), output_func(left))
            assert np.isclose(max(right, self_func(right)), output_func(right))
            assert np.isclose(max(mid, self_func(mid)), output_func(mid))

        return output

    """
    Computes min {x: f(x) >= x}
    """
    def compute_fixed_point(self, tol: float = 1e-12) -> float:
        output = None
        for i in range(len(self.changepoints)):
            a = self.changepoints[i-1] if i > 0 else 0
            b = self.changepoints[i]
            val_function = self.value_functions[i]
            if np.isclose(val_function(a), a, tol):
                output = a
                break
            if val_function(a) - a > tol and val_function(b) - b <= tol:
                output = val_function.b/(1 - val_function.a)
                break
        if output is None:
            print(self)
        assert output is not None
        return output
    