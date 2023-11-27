import numpy as np
import fun
import math
import random
import matplotlib.pyplot as plt

# We only discount every two episodes
def Bid(E,x,theta,gamma):
    L=math.floor(math.log(9*E**4-1)/math.log(1/math.sqrt(gamma)))+1
    a=random.random()
    if(a<L/E):
        bid=fun.Pi_Rand()
    else:
        bid=1+(x.T*theta).item()+2*random.random()-1
        noise=(2*random.random()-1)/E
        bid=bid+noise
        if(bid<0):
            bid=0
        elif(bid>3):
            bid=3
    return bid

def NPAC(K,theta,gamma,context):
    CDF_est=[]
    for i in range(201):
        CDF_est.append(0+i/200)
    theta_hat=np.matrix([[0],[0]])
    k=1
    bench_sum=0
    rev_sum=0
    revenue=[]
    benchmark=[]
    while(k<K+1):
        for i in range(1,K):
            E=math.floor(K**(1-2**(-i)))+1
            Phi=np.matrix([[0,0],[0,0]])
            Res=np.matrix([[0],[0]])
            context_record=[]
            bid_record=[]
            for j in range(E):
                x=context[k-1]
                context_record.append(x)
                Phi=Phi+x*x.transpose()
                a=random.random()
                if(a<1/E):
                    r=fun.Pi_Rand()
                else:
                    r=fun.UnPi(x,theta_hat,CDF_est)
                if(k%2==0):
                    r_opt=fun.Pi(x,theta)
                    bench_sum+=r_opt*(1-fun.CDF(r_opt-1-(x.T*theta).item()))
                    benchmark.append(bench_sum)
                else:
                    x_opt=np.matrix([[0],[1]])
                    r_opt=fun.Pi(x_opt,theta)
                    bench_sum+=r_opt*(1-fun.CDF(r_opt-1-(x_opt.T*theta).item()))
                    benchmark.append(bench_sum)
                rev_sum+=r*(1-fun.CDF(r-1-(x.T*theta).item()))
                revenue.append(rev_sum)
                bid=Bid(E,x,theta,gamma)
                bid_record.append(bid)
                Res=Res+x*(bid-1)
                k+=1
                if(k>K):
                    break
            theta_hat=Phi.I*Res
            CDF_est=fun.CDF_Update(bid_record,theta_hat,context_record,len(bid_record)+1)
            if(k>K):
                break
    plt.plot(revenue,label="NPAC-S")
    plt.plot(benchmark,label="Benchmark")
    plt.legend()
    plt.savefig("./NPAC.png")
    plt.close()
    regret=np.array(benchmark)-np.array(revenue)
    plt.plot(regret,label="Regret")
    plt.legend()
    plt.savefig("./NPAC_regret.png")
    plt.close()
    return regret


    
                
        
        
