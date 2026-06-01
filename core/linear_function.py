class LinearFunction:
    """
    f(x) = ax + b, with slope a and intercept b
    """
    def __init__(self, a: float, b: float) -> None:
        self.a = a
        self.b = b
    
    def __repr__(self):
        return f"{self.a:.4f} * x + {self.b:.4f}"
    
    def __call__(self, x: float) -> float:
        return self.a * x + self.b
    
    def derivative(self) -> "LinearFunction":
        return LinearFunction(0.0, self.a)
