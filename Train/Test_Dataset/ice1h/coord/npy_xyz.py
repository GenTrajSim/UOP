import numpy as np
data = np.load("../../ice1c/coord/5.5580.442.ice1c-3.npy")
out = open("test.xyz",'w')
print(len(data),file=out)
print("",file=out)
for line in range(len(data)):
    print ("O ", data[line][1], " ", data[line][2], " ", data[line][3], file=out)

