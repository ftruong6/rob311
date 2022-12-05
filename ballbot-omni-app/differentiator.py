
class Differentiator:
    def __init__(self):
        self.prev = 0.0
        self.dt = 0.005

    def reset(self,val):
        self.prev = val
    
    def differentiate(self,val):
        return (val-self.prev)/self.dt
