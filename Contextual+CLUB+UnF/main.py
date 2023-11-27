import CLUB
import numpy as np
import time
import SCORP
import NPAC
import matplotlib.pyplot as plt

if __name__ == '__main__':
    K=10000
    regret_CLUB_average=np.zeros(K)
    regret_SCORP_average=np.zeros(K)
    regret_NPAC_average=np.zeros(K)
    with open ("./result.txt", "w") as myfile:
        for i in range(30):
            context=CLUB.Gen_Context(K)
            time_start=time.time()
            regret_CLUB=CLUB.CLUB(K,np.matrix([[0.4],[0.6]]),0.9,context)
            time_end=time.time()
            print('totally cost for CLUB',time_end-time_start)
            time_start=time.time()
            regret_SCORP=SCORP.SCORP(K,np.matrix([[0.4],[0.6]]),0.9,context)
            time_end=time.time()
            print('totally cost for SCORP',time_end-time_start)
            time_start=time.time()
            regret_NPAC=NPAC.NPAC(K,np.matrix([[0.4],[0.6]]),0.9,context)
            time_end=time.time()
            print('totally cost for NPAC',time_end-time_start)
            print(regret_CLUB[K-1],regret_SCORP[K-1],regret_NPAC[K-1],file=myfile)
            regret_CLUB_average=(1-1/(i+1))*np.array(regret_CLUB_average)+1/(i+1)*np.array(regret_CLUB)
            regret_SCORP_average=(1-1/(i+1))*np.array(regret_SCORP_average)+1/(i+1)*np.array(regret_SCORP)
            regret_NPAC_average=(1-1/(i+1))*np.array(regret_NPAC_average)+1/(i+1)*np.array(regret_NPAC)
    plt.plot(regret_CLUB_average,label="CLUB")
    plt.plot(regret_SCORP_average,label="SCORP")
    plt.plot(regret_NPAC_average,label="NPAC")
    plt.legend()
    plt.savefig("./average.png")
    plt.close()
        

            