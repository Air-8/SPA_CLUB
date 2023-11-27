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
    reserve=[]
    # context=[]
    benchmark=[]
    bench_sum=0.0
    flag=0
    theta_hat=np.matrix([[0],[0]])
    while k<=K:
        # x=fun.Context()
        # context.append(x)
        x=context[k-1]
        a=random.random()
        bid=fun.Bid(x,theta,flag,K,gamma)
        r_opt=fun.Pi(x,theta)
        bench_sum+=r_opt*(1-fun.CDF(r_opt-1-(x.T*theta).item()))
        benchmark.append(bench_sum)
        if(a<1/K):
            r=fun.Pi_Rand()
        else:
            r=fun.Pi(x,theta_hat)
        # print(r,"0",k)
        rev_sum+=r*(1-fun.CDF(r-1-(x.T*theta).item()))
        if(bid>r):
            outcome.append(1)
        else:
            outcome.append(0)
        revenue.append(rev_sum)
        reserve.append(r)
        Phi=Phi+x*x.transpose()
        k+=1
        eigen=np.linalg.eigvals(2*np.linalg.inv(Phi)-np.linalg.inv(Phi_end))
        if(eigen.min()<0):
            flag=1
            for i in range(buffer):
                # x=fun.Context()
                x=context[k-1]
                r_opt=fun.Pi(x,theta)
                bench_sum+=r_opt*(1-fun.CDF(r_opt-1-(x.T*theta).item()))
                benchmark.append(bench_sum)
                # context.append(x)
                a=random.random()
                bid=fun.Bid(x,theta,flag,K,gamma)
                if(a<1/K):
                    r=fun.Pi_Rand()
                else:
                    r=fun.Pi(x,theta_hat)
                rev_sum+=r*(1-fun.CDF(r-1-(x.T*theta).item()))
                if(bid>r):
                    outcome.append(1)
                else:
                    outcome.append(0)
                revenue.append(rev_sum)
                reserve.append(r)
                Phi=Phi+x*x.transpose()
                # print(r,"1",k)
                k+=1
                if(k>K):
                    break
            theta_hat=fun.Update(outcome,context,reserve,k)
            flag=0
            Phi_end=Phi
    plt.plot(revenue,label="CLUB")
    plt.plot(benchmark,label="Benchmark")
    plt.legend()
    # plt.savefig("CLUB.png")
    plt.savefig("./CLUB.png")
    # plt.show()
    plt.close()
    regret=np.array(benchmark)-np.array(revenue)
    plt.plot(regret,label="Regret")
    plt.legend()
    plt.savefig("./CLUB_regret.png")
    # plt.show()
    plt.close()
    return regret
    
# CLUB(1000,np.matrix([[0.4],[0.6]]),0.9)

    
                
            

            
           
            
            

