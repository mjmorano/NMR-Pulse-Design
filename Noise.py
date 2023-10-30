import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from joblib import Parallel, delayed
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import rk5
from time import time
from numba import jit

@jit(nopython=True)
def calc_sincos_window(t,params,t_max):
    B = 1e-7*np.sin(np.pi*t/t_max)*np.sin(params[0]*t)*(params[1]*np.cos(B0*gamma_n*t+params[2]*np.cos(params[3]*t)) + params[4]*np.cos(B0*gamma_3*t+params[5]*np.cos(params[6]*t)))
    return B

@jit(nopython=True)
def bloch_noise(gamma,B0,B1,t,y):
    return np.array([gamma*B0*y[1], gamma*B1*y[2]-gamma*B0*y[0], -gamma*B1*y[1]])

@jit(nopython=True)
def bloch(t,y,gamma,B,params,t_max):
    B1 = calc_sincos_window(t,params,t_max)
    return [gamma*B*y[1], gamma*B1*y[2]-gamma*B*y[0], -gamma*B1*y[1]]

def calc_M_final(B0, B1, time, int_times, dt):
    SNR = 80
    noise_amp = B1.max()*10**(-SNR/20)
    noise = np.random.normal(scale=noise_amp,size=B1.size)
    B1 += noise
    B1 = interp1d(time,B1,kind='cubic')
    B1 = B1(int_times)

    sol_n = rk5.rk5(bloch_noise, int_times, dt, np.array([0.0,0.0,1.0]), gamma_n, B0, B1)
    sol_3 = rk5.rk5(bloch_noise, int_times, dt, np.array([0.0,0.0,1.0]), gamma_3, B0, B1)

    dot = np.dot(sol_n[-1,:], sol_3[-1,:])
    dot = dot / np.sqrt(sol_n[-1,:].dot(sol_n[-1,:])) / np.sqrt(sol_3[-1,:].dot(sol_3[-1,:]))
    # print(np.arccos(dot))
    return np.arccos(dot)

    # return sol_n[-1,0],sol_n[-1,1],sol_n[-1,2],sol_3[-1,0],sol_3[-1,1],sol_3[-1,2]

# def calc_phase(Mx_n,My_n,Mx_He,My_He):
#     vector_1 = [Mx_n, My_n]
#     vector_2 = [Mx_He, My_He]

#     unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
#     unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
#     dot_product = np.dot(unit_vector_1, unit_vector_2)
#     angle = np.arccos(dot_product)

#     return angle

params = np.array([10.43601267, 10.99323644, 2.04251651, -1.76507477, -10.40391301, -0.50737193, -1.95364846])
tau = 100e-3              # Pulse length
fs = 10000
times = np.linspace(0., tau, int(fs*tau))
dt = 1e-5
int_times = np.linspace(0. , tau,int(tau/dt))
gamma_n = -1.83247171e8          # Neutron gyromagnetic ratio in rad/s*T
gamma_3 = -2.037894569e8         # He3 gyromagnetic ratio in rad/s*T
B0 = 3e-6
# sol = solve_ivp(bloch, [0.,tau], [0., 0., 1.], args=(gamma_n, B0, params, tau), method='LSODA', rtol=1e-10, atol=1e-10)
# no_noise = sol.y[:,-1]
# print(no_noise)
B1 = calc_sincos_window(times, params, tau)
B0 = B0*np.ones(int_times.size)

num_runs = 1000

start = time()
z = np.array(Parallel(n_jobs=7)(delayed(calc_M_final)(B0,B1,times,int_times,dt) for i in range(num_runs)))

# print(z.shape)
# phi = np.arctan2(z[:,1], z[:,0])
# print('Mean phi = {:.5f}'.format(phi.mean()))
# phi -= phi.mean()
# phi *= 1e3
# theta = np.arctan2(z[:,2],np.sqrt(z[:,0]**2 + z[:,1]**2))
# print('Mean theta = {:.5f}'.format(theta.mean()))
# theta -= theta.mean()
# theta *= 1e3
print(np.std(z)*1e3)

plt.hist(z,bins=101)
plt.show()

end = time()
print("Elapsed time: " + str(end-start) + " seconds")

# _, bins, _ = plt.hist(phi,101,density=1)
# mu, sigma = norm.fit(phi)
# fit = norm.pdf(bins,mu,sigma)
# print("mu = {:5e}".format(mu))
# print("sigma - {:5e}".format(sigma))

# plt.plot(bins,fit)
# plt.xlabel('Mean Subtracted Phase (mrad)')
# plt.ylabel('Normalized Counts')
# plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# plt.show()

# _, bins, _ = plt.hist(theta,101,density=1)
# mu, sigma = norm.fit(theta)
# fit = norm.pdf(bins,mu,sigma)
# print("mu = {:5e}".format(mu))
# print("sigma - {:5e}".format(sigma))

# plt.plot(bins,fit)
# plt.xlabel('Mean Subtracted Phase (mrad)')
# plt.ylabel('Normalized Counts')
# plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# plt.show()
