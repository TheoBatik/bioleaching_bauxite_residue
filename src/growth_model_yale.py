#39;#39;#39;A. niger
# Luideking Piret + Contois growth
# Considers glucose and oxygen as substrate#39;#39;#39;
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

#39;#39;#39;Initialisation X- biomass, DO - dissolved oxygen, S- substrate
# (glucose), P - gluconic acid #39;#39;#39;
X0 = 0.04 #g/L
DO0 = 0.0065 #g/L
S0 = 150 #g/L
P0 = 0 #g/L
#39;#39;#39;Mass balance equations from Znad paper. Values for fitting parameters
# taken directly#39;#39;#39;
umax = 0.361 # /h
Ks = 21.447 # g/L
Ko = 0.001061
alpha = 2.58
beta = 0.1704 # /h
q = 2.17680
r = 0.29370
s = 0.2724
v = 0.0425
Mo = 16 # g/mol molar mass oxygen
Mga = 196.16 # g/mol molar mass gluconic acid
kla = 63 # h-1
csat = 0.0065 # guess
tx = np.array([0.181818, 3.09091, 7.09091, 10, 13.0909, 16.1818, 19.0909,
22.1818, 25.0909, 28.1818, 31.0909, 34.1818, 37.0909, 40.5455, 43.4545,
46.5455, 49.4545, 52.5455, 54.5455])
Xy = np.array([0.0725806, 0.326613, 1.19758, 2.28629, 2.97581, 4.1371,
4.57258, 6.13306, 5.95161, 6.75, 7.33065, 6.89516, 6.60484, 7.1129,
7.33065, 7.22177, 7.33065, 7.47581, 7.72984])
Sy = np.array([150.539, 144.564, 142.573, 135.602, 132.614, 126.141,
127.137, 123.154, 117.676, 114.689, 109.71, 100.747, 88.2988, 83.8174,
84.3154, 73.8589, 60.9129, 49.4606, 45.9751])
Py = np.array([0, 1.83406, 4.58515, 7.64192, 9.47598, 17.1179, 19.2576,
17.4236, 26.8996, 30.262, 35.7642, 39.738, 43.4061, 48.6026, 51.3537,
56.2445, 57.7729, 60.8297, 68.7773])

def MICROBIAL(f,t):
    X = f[0]
    S = f[1]
    DO = f[2]
    P = f[3]
    u = umax*(S/(Ks*X+S))*(DO/(Ko*X+DO))
    ddt0 = u*X #dXdt
    rs = -q * ddt0 - r * X
    ddt1 = rs #dSdt
    ddt2 = kla*(csat - DO) - s*ddt0 - v*X
    rp = alpha * ddt0 + beta * X
    ddt3 = rp #dPdt
    ddt = [ddt0, ddt1, ddt2, ddt3]
    return ddt

b0 = [X0, S0, DO0, P0]
t = np.linspace(0.01,55,55)
t0 = [0,0]
g = odeint(MICROBIAL,b0,t)
cX = g[:,0]
cS = g[:,1]
cDO = g[:,2]
cP = g[:,3]
quot;quot;quot;plt.figure(1)
plt.plot(t,cX,#39;r#39;)
plt.plot(t,cS,#39;b#39;)
#plt.plot(t,cP,#39;g#39;)
plt.xlabel(#39;Time (days)#39;)
plt.ylabel(#39;Concentration (g/L)#39;)
plt.legend([#39;X#39;,#39;S#39;])quot;quot;quot;
plt.figure(1)
plt.plot(tx, Xy, #39;o#39;)
plt.plot(t,cX,#39;g#39;)
plt.xlabel(#39;Time (hours)#39;)
plt.ylabel(#39;Biomass Concentration (g/L)#39;)
#plt.legend([#39;P#39;])
plt.figure(2)
plt.plot(tx, Py, #39;o#39;)
plt.plot(t,cP,#39;g#39;)
plt.xlabel(#39;Time (hours)#39;)
plt.ylabel(#39;Gluconic Acid Concentration (g/L)#39;)
#plt.legend([#39;P#39;])
plt.figure(3)
plt.plot(tx, Sy, #39;o#39;)
plt.plot(t,cS,#39;g#39;)
plt.xlabel(#39;Time (hours)#39;)
plt.ylabel(#39;Glucose Concentration (g/L)#39;)
#plt.legend([#39;P#39;])
plt.figure(4)
plt.plot(t,cDO,#39;g#39;)
plt.xlabel(#39;Time (hours)#39;)
plt.ylabel(#39;DO Concentration (g/L)#39;)
plt.show()