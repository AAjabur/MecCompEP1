import numpy as np
from numpy import pi
from utils.rk_4 import rk_4_solve, analyze_rk_4_error

from matplotlib import pyplot as plt

class CarVecFunctionGenerator:
    def __init__(
        self,
        M = 1783,           # kg
        f_e = 2100,         # rpm
        m_e = 20,           # kg
        r = 0.045,          # m
        V = 50,             # km/h
        k1 = 2.8 * 10**7,   # N/m
        k2 = 2.8 * 10**7,   # N/m
        c1 = 3 * 40**4,     # kg/s
        c2 = 3 * 40**4,     # kg/s
        a = 1220,           # mm
        b = 1500,           # mm
        Ic = 4000,          # kg m^2
        e = 0.75,           # m
        L = 0.5,            # m
        A = 60,             # mm
        f = 0.35            # m
    ):
        self.M = M
        self.f_e = f_e / 60
        self.m_e = m_e
        self.r = r
        self.V = V / 3.6
        self.k1 = k1
        self.k2 = k2
        self.c1 = c1
        self.c2 = c2
        self.a = a / 1000
        self.b = b / 1000
        self.Ic = Ic
        self.e = e
        self.L = L
        self.A = A / 1000
        self.f = f
        self.omega = 2 * pi * self.V / self.L
        self.omega_e = 2 * pi * self.f_e
        self.Fn = self.m_e * (self.omega_e ** 2) * self.r

    def d1(self, t):
        if t > 2:
            return 0
        
        return self.A*(1 - np.cos(self.omega * t))

    def dd1(self, t):
        if t > 2:
            return 0
        
        return self.A*self.omega*np.sin(self.omega*t)

    def d2(self, t):
        if t > 2:
            return 0
        
        return self.A*(1 + np.cos(self.omega * t))

    def dd2(self, t):
        if t > 2:
            return 0
        
        return -self.A*self.omega*np.sin(self.omega * t)
    
    def car_vec_func(self, t: float, x_vec: np.array):
        y_vec = np.zeros(4)
        x = x_vec[0]
        dx = x_vec[1]
        theta = x_vec[2]
        dtheta = x_vec[3]

        y_vec[0] = x_vec[1]
        y_vec[1] = -self.k1 * (x - self.a*theta - self.d1(t)) - self.k2*(x + self.b*theta - self.d2(t)) - self.c1*(dx - self.a*dtheta - self.dd1(t)) - self.c2*(dx + self.b*dtheta - self.dd2(t)) + self.Fn*np.sin(self.omega_e*t)
        y_vec[1] /= self.M
        y_vec[2] = x_vec[3]
        y_vec[3] = self.k1*(x - self.a*theta - self.d1(t))*self.a - self.k2*(x + self.b*theta - self.d2(t))*self.b + self.c1*(dx - self.a*dtheta - self.dd1(t))*self.a - self.c2*(dx + self.b*dtheta - self.dd2(t))*self.b - self.Fn*np.sin(self.omega_e*t)*self.e - self.Fn*np.cos(self.omega_e*t)*self.f
        y_vec[3] /= self.Ic
        
        return y_vec
