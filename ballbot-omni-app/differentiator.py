
import numpy as np
class Differentiator:
    def __init__(self):
        self.prev = 0.0
        self.dt = 0.005

    def reset(self,val):
        self.prev = val
    
    def differentiate(self,val):
        result = (val-self.prev)/self.dt
        self.prev = val
        return result

class SlewRateLimiter:
    def __init__(self):
        self.dt = 0.005
        self.maxDerivative = 0.1
        self.prev = 0
    
    def limit(self,val):
        dval = (val - self.prev)/self.dt
        result = val
        if(np.abs(dval)>self.maxDerivative):
            result = np.sign(dval)*self.maxDerivative*self.dt+self.prev
        self.prev = result
        return result

    def reset(self,val):
        self.differentiator.reset(self,val)

if __name__ == "__main__":
    limiter = SlewRateLimiter()

    for x in range(0,1000):
        y = limiter.limit(x)
        print(x)
        print(y)

