import numpy as np
import random
import fun
from math import log
import math
import matplotlib.pyplot as plt

def Gen_Context(K):
    context=[]
    for i in range(K):
        context.append(fun.Context())
    return context
    
def CLUB(K,theta,gamma,context):
    buffer=int(math.floor(3*log(K)/log(1/gamma)))+1
    k=1
    Phi=np.identity(2)
    Phi_end=np.identity(2)
    revenue=[]
    rev_sum=0.0
    outcome=[]
    non_buffer=[]
    outcome_simul=[]
    reserve=[]
    bid_record=[]
    benchmark=[]
    bench_sum=0.0
    flag=0
    theta_hat=np.matrix([[0],[0]])
    # 201 points for -1 to 1
    CDF_est=[]
    for i in range(201):
        CDF_est.append(0+i/200)
    while k<=K:
        x=context[k-1]
        a=random.random()
        bid=fun.Bid(x,theta,flag,K,gamma)
        bid_record.append(bid)
        non_buffer.append(x)
        r_opt=fun.Pi(x,theta)
        bench_sum+=r_opt*(1-fun.CDF(r_opt-1-(x.T*theta).item()))
        benchmark.append(bench_sum)
        if(a<1/K):
            r=fun.Pi_Rand()
        else:
            r=fun.UnPi(x,theta_hat,CDF_est)
        # print(r,r_opt,"0",k)
        rev_sum+=r*(1-fun.CDF(r-1-(x.T*theta).item()))
        if(bid>r):
            outcome.append(1)
        else:
            outcome.append(0)
        r_simul=fun.Pi_Rand()
        if(bid>r_simul):
            outcome_simul.append(1)
        else:
            outcome_simul.append(0)
        revenue.append(rev_sum)
        reserve.append(r)
        Phi=Phi+x*x.transpose()
        k+=1
        eigen=np.linalg.eigvals(2*np.linalg.inv(Phi)-np.linalg.inv(Phi_end))
        if(eigen.min()<0):
            flag=1
            for i in range(buffer):
                x=context[k-1]
                r_opt=fun.Pi(x,theta)
                bench_sum+=r_opt*(1-fun.CDF(r_opt-1-(x.T*theta).item()))
                benchmark.append(bench_sum)
                a=random.random()
                bid=fun.Bid(x,theta,flag,K,gamma)
                # bid_record.append(bid)
                if(a<1/K):
                    r=fun.Pi_Rand()
                else:
                    r=fun.UnPi(x,theta_hat,CDF_est)
                # print(r,r_opt,"1",k)
                rev_sum+=r*(1-fun.CDF(r-1-(x.T*theta).item()))
                if(bid>r):
                    outcome.append(1)
                else:
                    outcome.append(0)
                r_simul=fun.Pi_Rand()
                if(bid>r_simul):
                    outcome_simul.append(1)
                else:
                    outcome_simul.append(0)
                revenue.append(rev_sum)
                reserve.append(r)
                Phi=Phi+x*x.transpose()
                k+=1
                if(k>K):
                    break
            theta_hat=fun.UnUpdate(outcome_simul,context,k)
            CDF_est=fun.CDF_Update(bid_record,theta_hat,non_buffer,len(non_buffer)+1)
            flag=0
            Phi_end=Phi
    plt.plot(revenue,label="CLUB")
    plt.plot(benchmark,label="Benchmark")
    plt.legend()
    plt.savefig("./CLUB.png")
    plt.close()
    regret=np.array(benchmark)-np.array(revenue)
    plt.plot(regret,label="Regret")
    plt.legend()
    plt.savefig("./CLUB_regret.png")
    plt.close()
    return regret
    
if __name__=='__main__':
    K=10000
    context=Gen_Context(K)
    regret_CLUB=CLUB(K,np.matrix([[0.4],[0.6]]),0.9,context)
    print(regret_CLUB[K-1])
    
                
            

            
           
            
            

