import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

#Constantes utiles
N = 1024
visc = 1
L = 100
dt = 0.05
n_t = int(200/0.05)

#Vecteurs utiles
x = np.linspace(0,100,N)
t = np.linspace(0,200,n_t)
k = np.arange(-N/2, N/2) 

#Conditions initiales
CI = np.cos(2*np.pi*x/L) + 0.1*np.cos(4*np.pi*x/L)

#Matrices de solutions, réelle et esp.spectral
u = np.zeros((n_t,N)) #40K lignes, 1024 colonnes
u_spec = np.zeros((n_t, N))

#On remplit les matrices avec les C.I.
u[0] = CI
u_spec[0] = 1/N * np.fft.fft(np.fft.fftshift(CI))

#Transfo de Fourier de la partie lin.
f_L = (2*np.pi*k/L)**2 - visc*(2*np.pi*k/L)**4

#On remplit le reste des matrices
a = 1j * 2*np.pi/L * k #Pour appliquer d/dx

u_spec[1] = (1 + dt*0.5*f_L)/(1 - dt*0.5*f_L)*u_spec[0] -  (3/2*(a*u_spec[0]**2) - 1/2*(a*u_spec[0]**2)) * dt/(1 - dt*0.5*f_L)
for i in range(1,n_t - 1):
    u_spec[i+1] = (1 + dt*0.5*f_L)/(1 - dt*0.5*f_L)*u_spec[i] -  (3/2*(a*u_spec[i]**2) - 1/2*(a*u_spec[i-1]**2)) * dt/(1 - dt*0.5*f_L)
    #fftshift
    u[i+1] = np.fft.ifft(np.fft.ifftshift(u_spec[i+1]))
    
#On affiche nos résultats
