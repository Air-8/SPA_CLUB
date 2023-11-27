import numpy as np
import math
import fun
import random
import matplotlib.pyplot as plt

def Bid(K,x,theta,i,explore,gamma):
    untruthful=math.sqrt(6*K*gamma**(explore-i)*3/(1-gamma))
    bid=1+(x.T*theta).item()+2*random.random()-1+untruthful*(2*random.random()-1)
    if(bid<0):
        bid=0
    elif(bid>3):
        bid=3
    return bid

def SCORP(K,theta,gamma,context):  
    outcome=[]
    bench_sum=0
    rev_sum=0
    revenue=[]
    benchmark=[]
    explore=math.floor(K**(2/3))+1
    for i in range(explore):
        x=context[i]
        r=fun.Pi_Rand()
        r_opt=fun.Pi(x,theta)
        bench_sum+=r_opt*(1-fun.CDF(r_opt-1-(x.T*theta).item()))
        benchmark.append(bench_sum)
        bid=Bid(K,x,theta,i,explore,gamma)
        if(bid>r):
            outcome.append(1)
        else:
            outcome.append(0)
        rev_sum+=r*(1-fun.CDF(r-1-(x.T*theta).item()))
        revenue.append(rev_sum)
    theta_hat=fun.UnUpdate(outcome,context,explore+1)
    for i in range(explore,K):
        x=context[i]
        r=fun.Pi(x,theta_hat)
        r_opt=fun.Pi(x,theta)
        bench_sum+=r_opt*(1-fun.CDF(r_opt-1-(x.T*theta).item()))
        benchmark.append(bench_sum)
        rev_sum+=r*(1-fun.CDF(r-1-(x.T*theta).item()))
        revenue.append(rev_sum)
    plt.plot(revenue,label="SCORP")
    plt.plot(benchmark,label="Benchmark")
    plt.legend()
    plt.savefig("./SCORP.png")
    plt.close()
    regret=np.array(benchmark)-np.array(revenue)
    plt.plot(regret,label="Regret")
    plt.legend()
    plt.savefig("./SCORP_regret.png")
    plt.close()
    return regret
        
