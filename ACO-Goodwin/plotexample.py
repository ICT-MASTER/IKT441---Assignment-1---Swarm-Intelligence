import matplotlib.pyplot as plt



data1 = [int(i.split()[1]) for i in open("evap.csv").readlines()] 
data2 = [int(i.split()[1]) for i in open("mmas.csv").readlines()] 
#data1 = [1,1,1,1,3,3,3,3,2,2,2]
#data2 = [2,2,2,2,2,2,4,4,4,3,3]
plt.plot(data1,label="evap")
plt.plot(data2,label="mmas")
plt.legend()
plt.show()

