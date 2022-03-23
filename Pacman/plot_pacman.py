#Plot load-displacement curve#
import matplotlib.pyplot as plt
X1,Y1,Z1,W1 = [],[],[],[]
for line in open('ForceDisp_pacman_energy.txt','r'):
    values1 = [float(s) for s in line.split()]
    X1.append(values1[0])
    Y1.append(values1[1])
    Z1.append(values1[2])
    W1.append(values1[3])

plt.figure()
plt.plot(X1,Y1,'b-',label='Elastic energy')
plt.plot(X1,Z1,'r-',label='Dissipated energy')
plt.plot(X1,W1,'k--',label='Total energy')
#
plt.xlim([1.4,1.75])
#
plt.title('Energy vs t')
plt.xlabel('t')
plt.ylabel('Energy')
plt.legend()
plt.show()
