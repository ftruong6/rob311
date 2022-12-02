import numpy as np
from DataLogger import dataLogger

class LPFS:
    def __init__(self):
        self.x1prev = 0.0
        self.x2prev = 0.0
        self.Kp = 1.0
        self.Kd = 1.0
        self.F = 1.0
        self._rprev = 0.0
        self.dt = 0.005

    def reset(self,r):
        self.x1prev = r #when first start to use, set the filter value to equal initial observation
        self.x2prev = 0.0
        self._rprev = r 
    #r is the process variable.
    # function call calculates the raw 
    def __calcDr(self,r):
        result = (r-self._rprev) #/self.dt
        self._rprev = r
        return result

    def __sat(self,input):
        result = np.max([-(self.F),np.min([self.F,input])])
        return result

    def __calcB(self,r):
        dr = self.__calcDr(r) 
        B = (self.Kd*(dr-self.x2prev)+self.Kp*(r+self.dt*self.x2prev-self.x1prev))/(1.0+self.dt*self.Kd+(self.dt**2.0)*self.Kp)
        return B

    # updates the filter with a new observation r
    # returns the filtered value of r.
    def filter(self,r):
        x2 = self.dt*self.__sat(self.__calcB(r))+self.x2prev #reread research paper if you don't understand this
        x1 = self.dt*self.x2prev+self.x1prev
        self.x2prev = x2
        self.x1prev = x1
        return x1

    # sets the gains to estimates according to rule of thumb
    # takes two arguments
    def estimateGains(self,w,A):
        self.Kd = 60.0*w
        self.Kp = self.Kd*w*0.2
        self.F = 8.0*A/(np.pi**2.0)*(w**2.0)
        return self.Kd, self.Kp, self.F
    

if __name__ == "__main__":
    data = []
    filtered = []
    dt = 0.005

    lpfs = LPFS()
    lpfs.dt= dt
    lpfs.estimateGains(40.0,1.0)
    lpfs.reset(0)

    filename = 'LPFS test'
    dl = dataLogger(filename + '.txt')
    for i in range(10000):
        r = np.sin(i*dt*2.0*np.pi/2)
        data = [r]+[lpfs.filter(r)]
        dl.appendData(data)
    dl.writeOut()
    pass
