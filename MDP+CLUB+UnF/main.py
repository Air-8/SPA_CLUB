import CLUB
import numpy as np
import time
import NPAC
import matplotlib.pyplot as plt
from math import log

if __name__ == '__main__':
    K,H=10000,2
    poly1=H*log(K)**2
    poly2=H**2*log(K)**4
    regret_CLUB_average=np.zeros(2*K)
    regret_NPAC_average=np.zeros(2*K)
    with open ("./result.txt", "w") as myfile:
        for i in range(30):
            context=CLUB.Gen_Context(K)
            context1=CLUB.Gen_Context(K)
            context_all=[]
            for j in range(K):
                context_all.append(context[j])
                context_all.append(context1[j])
            time_start=time.time()
            regret_NPAC=NPAC.NPAC(2*K,np.matrix([[0.4],[0.6]]),0.9,context_all)
            time_end=time.time()
            print('totally cost for NPAC-S',time_end-time_start)
            regret_NPAC_average=(1-1/(i+1))*np.array(regret_NPAC_average)+1/(i+1)*np.array(regret_NPAC)
            time_start=time.time()
            regret_CLUB=CLUB.CLUB(K,H,np.matrix([[0.4],[0.6]]),0.9,context,poly1,poly2)
            time_end=time.time()
            print('totally cost for CLUB',time_end-time_start)
            regret_CLUB_average=(1-1/(i+1))*np.array(regret_CLUB_average)+1/(i+1)*np.array(regret_CLUB)
            print(regret_CLUB[2*K-1],regret_NPAC[2*K-1],file=myfile)
    plt.plot(regret_CLUB_average,label="CLUB")
    plt.plot(regret_NPAC_average,label="NPAC-S")
    plt.legend()
    plt.savefig("./average.png")
    plt.close()    
    

        

            