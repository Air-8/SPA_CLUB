import CLUB
import numpy as np
import time

if __name__ == '__main__':
    time_start=time.time()
    K=10000
    context=CLUB.Gen_Context(K)
    regret_CLUB=CLUB.CLUB(K,np.matrix([[0.4],[0.6]]),0.9,context)
    # print("CLUB regret:",regret_CLUB)
    time_end=time.time()
    print('totally cost',time_end-time_start)