import numpy as np
import random
import fun
from math import log
from math import sqrt
import math
import matplotlib.pyplot as plt

def Gen_Context(K):
    context=[]
    for i in range(K):
        context.append(fun.Context())
    return context

def Bid(x,theta,flag,K,H,gamma):
    scale=3*H*sqrt(2)/K/sqrt(1-gamma)
    val=1+(theta.T*x).item()+2*random.random()-1
    if (flag==0):
        bid=val+scale*(2*random.random()-1)
        if(bid<0):
            bid=0
        elif(bid>3):
            bid=3
    else:
        bid=3*random.random()
    return bid
    
    #two actions, the first one leads to context 1 for sure and the second one leads to context 2 for sure
def CLUB(K,H,theta,gamma,context,poly1,poly2):
    buffer=int(math.floor(3*log(K)/log(1/gamma)))+1
    k=1
    action=0
    a=random.random()
    if a<=0.5:
        action=1
    Phi_1=np.identity(4)
    # print(Phi_1)
    Phi_2=np.identity(4)  
    Phi_end_1=np.identity(4)
    Phi_end_2=np.identity(4)   
    phi_record=[]
    phi1_record=[]
    phi2_record=[]
    bid_record=[]   
    benchmark=[]
    revenue=[]
    benchmark_sum=0
    revenue_sum=0
    theta_hat=np.matrix([[0],[0]])
    Q = [[0, 0, 0, 0],
        [0, 0, 0, 0]]
    CDF_est=[]
    for i in range(201):
        CDF_est.append(0+i/200)
    non_buffer=[]   
    x_record=[]  
    outcome_simul=[] 
    while k<=K:
        # print(k)
        for h in range(1,H+1):
            # print("h",h)
            if(h==1):
                x=context[k-1]
            else:
                if action==0:
                    x=np.matrix([[1],[0]])
                else:
                    x=np.matrix([[0],[1]])
            non_buffer.append(x)
            x_record.append(x)
            bid=Bid(x,theta,0,K,H,gamma)
            bid_record.append(bid)
            r_simul=fun.Pi_Rand()
            if(bid>r_simul):
                outcome_simul.append(1)
            else:
                outcome_simul.append(0)
            if h==1:
                r_opt=fun.Pi(x,theta)
                benchmark_sum+=r_opt*(1-fun.CDF(r_opt-1-(x.T*theta).item()))
                benchmark.append(benchmark_sum)
            else:
                x_opt=np.matrix([[0],[1]])
                r_opt=fun.Pi(x_opt,theta)
                benchmark_sum+=r_opt*(1-fun.CDF(r_opt-1-(x_opt.T*theta).item()))
                benchmark.append(benchmark_sum)
            a=random.random()
            if a<=1/(H*K):
                r=fun.Pi_Rand()
                b=random.random()
                if b<=0.5:
                    action=0
                    y1,y2=1.0,0.0
                else:
                    action=1
                    y1,y2=0.0,1.0
                phi=np.matrix([x[0],x[1],[y1],[y2]])
                # print(phi)
                phi_record.append(phi)
                if(h==1):
                    Phi_1=Phi_1+phi.transpose()*phi
                    phi1_record.append(phi)
                else:
                    Phi_2=Phi_2+phi.transpose()*phi 
                    phi2_record.append(phi)
                revenue_sum+=r*(1-fun.CDF(r-1-(x.T*theta).item()))
                revenue.append(revenue_sum)
            else:
                r=fun.UnPi(x,theta_hat,CDF_est)
                # choose action
                if h==0:
                    if x[0]==1:
                        if Q[0][0]>Q[0][1]:
                            action=0
                            y1,y2=1.0,0.0
                        else:
                            action=1
                            y1,y2=0.0,1.0
                    else:
                        if Q[0][2]>Q[0][3]:
                            action=0
                            y1,y2=1.0,0.0
                        else:
                            action=1
                            y1,y2=0.0,1.0
                else:
                    if x[0]==1:
                        if Q[1][0]>Q[1][1]:
                            action=0
                            y1,y2=1.0,0.0
                        else:
                            action=1
                            y1,y2=0.0,1.0
                    else:
                        if Q[1][2]>Q[1][3]:
                            action=0
                            y1,y2=1.0,0.0
                        else:
                            action=1
                            y1,y2=0.0,1.0
                phi=np.matrix([x[0],x[1],[y1],[y2]])
                # print(phi)
                phi_record.append(phi)
                if(h==1):
                    Phi_1=Phi_1+phi.transpose()*phi
                    phi1_record.append(phi)
                else:
                    Phi_2=Phi_2+phi.transpose()*phi
                    phi2_record.append(phi)
                revenue_sum+=r*(1-fun.CDF(r-1-(x.T*theta).item()))
                revenue.append(revenue_sum)
        k+=1
        eigen_1=np.linalg.eigvals(2*np.linalg.inv(Phi_1.astype(float))-np.linalg.inv(Phi_end_1.astype(float)))
        eigen_2=np.linalg.eigvals(2*np.linalg.inv(Phi_2.astype(float))-np.linalg.inv(Phi_end_2.astype(float)))
        if eigen_1.min()<0 or eigen_2.min()<0 or (k & (k - 1)) == 0:
            # print("update")
            for i in range(buffer):
                for h in range(1,H+1):
                    if(h==1):
                        x=context[k-1]
                    else:
                        if action==0:
                            x=np.matrix([[1],[0]])
                        else:
                            x=np.matrix([[0],[1]])
                    x_record.append(x)
                    bid=Bid(x,theta,1,K,H,gamma)
                    r_simul=fun.Pi_Rand()
                    if(bid>r_simul):
                        outcome_simul.append(1)
                    else:
                        outcome_simul.append(0)
                    if h==1:
                        r_opt=fun.Pi(x,theta)
                        benchmark_sum+=r_opt*(1-fun.CDF(r_opt-1-(x.T*theta).item()))
                        benchmark.append(benchmark_sum)
                    else:
                        x_opt=np.matrix([[0],[1]])
                        r_opt=fun.Pi(x_opt,theta)
                        benchmark_sum+=r_opt*(1-fun.CDF(r_opt-1-(x_opt.T*theta).item()))
                        benchmark.append(benchmark_sum)
                    a=random.random()
                    if a<=1/(H*K):
                        r=fun.Pi_Rand()
                        b=random.random()
                        if b<=0.5:
                            action=0
                            y1,y2=1,0
                        else:
                            action=1
                            y1,y2=0,1
                        phi=np.matrix([x[0],x[1],[y1],[y2]])
                        phi_record.append(phi)
                        if(h==1):
                            Phi_1=Phi_1+phi.transpose()*phi
                            phi1_record.append(phi)
                        else:
                            Phi_2=Phi_2+phi.transpose()*phi
                            phi2_record.append(phi)
                        revenue_sum+=r*(1-fun.CDF(r-1-(x.T*theta).item()))
                        revenue.append(revenue_sum)
                    else:
                        r=fun.UnPi(x,theta_hat,CDF_est)
                        # choose action
                        if h==0:
                            if x[0]==1:
                                if Q[0][0]>Q[0][1]:
                                    action=0
                                    y1,y2=1.0,0.0
                                else:
                                    action=1
                                    y1,y2=0.0,1.0
                            else:
                                if Q[0][2]>Q[0][3]:
                                    action=0
                                    y1,y2=1.0,0.0
                                else:
                                    action=1
                                    y1,y2=0.0,1.0
                        else:
                            if x[0]==1:
                                if Q[1][0]>Q[1][1]:
                                    action=0
                                    y1,y2=1.0,0.0
                                else:
                                    action=1
                                    y1,y2=0.0,1.0
                            else:
                                if Q[1][2]>Q[1][3]:
                                    action=0
                                    y1,y2=1.0,0.0
                                else:
                                    action=1
                                    y1,y2=0.0,1.0
                        phi=np.matrix([x[0],x[1],[y1],[y2]])
                        # print("phi",context[k-1],phi)
                        phi_record.append(phi)
                        if(h==1):
                            Phi_1=Phi_1+phi.transpose()*phi
                            phi1_record.append(phi)
                        else:
                            Phi_2=Phi_2+phi.transpose()*phi
                            phi2_record.append(phi)
                        revenue_sum+=r*(1-fun.CDF(r-1-(x.T*theta).item()))
                        revenue.append(revenue_sum)
                k+=1
                # print(k)
                if(k>K):
                    break
            theta_hat=fun.UnUpdate(outcome_simul,x_record,len(x_record)+1)
            CDF_est=fun.CDF_Update(bid_record,theta_hat,non_buffer,len(non_buffer)+1)
            Phi_end_1=Phi_1
            Phi_end_2=Phi_2
            sum1=np.matrix([[0],[0],[0],[0]])
            R1=[0,0]
            # print("theta_hat",theta_hat)
            R1[0]=fun.UnPi_rev(np.matrix([[1],[0]]),theta_hat,CDF_est)
            R1[1]=fun.UnPi_rev(np.matrix([[0],[1]]),theta_hat,CDF_est)   
            # print("R1",R1[0])
            phi=np.matrix([[1],[0],[1],[0]])
            Q[1][0]=min(6,(R1[0]+poly1*(phi.T*np.linalg.inv(Phi_2.astype(float))*phi)+poly2/sqrt(k)).item())
            phi=np.matrix([[1],[0],[0],[1]])
            Q[1][1]=min(6,(R1[1]+poly1*(phi.T*np.linalg.inv(Phi_2.astype(float))*phi)+poly2/sqrt(k)).item())
            phi=np.matrix([[0],[1],[1],[0]])
            Q[1][2]=min(6,(R1[0]+poly1*(phi.T*np.linalg.inv(Phi_2.astype(float))*phi)+poly2/sqrt(k)).item())
            phi=np.matrix([[0],[1],[0],[1]])
            Q[1][3]=min(6,(R1[1]+poly1*(phi.T*np.linalg.inv(Phi_2.astype(float))*phi)+poly2/sqrt(k)).item())       
            for i in range(len(phi1_record)):
                a,b=0,0
                if(phi1_record[i][0]==1):
                    a,b=Q[1][0],Q[1][1]
                else:
                    a,b=Q[1][2],Q[1][3]
                # print("sum",sum1)
                # print(phi1_record[i])
                sum1=sum1+phi1_record[i]*max(a,b) 
            # print("sum",sum1)
            # print("Phi",Phi_1)  
            w1=np.linalg.inv(Phi_1.astype(float))*sum1
            # print(np.linalg.inv(Phi_1.astype(float)))
            # print("w1",w1)
            phi=np.matrix([[1],[0],[1],[0]])
            Q[0][0]=min(6,(w1.T*phi+R1[0]+poly1*(phi.T*np.linalg.inv(Phi_1.astype(float))*phi)+poly2/sqrt(k)).item())
            phi=np.matrix([[1],[0],[0],[1]])
            Q[0][1]=min(6,(w1.T*phi+R1[1]+poly1*(phi.T*np.linalg.inv(Phi_1.astype(float))*phi)+poly2/sqrt(k)).item())
            phi=np.matrix([[0],[1],[1],[0]])
            Q[0][2]=min(6,(w1.T*phi+R1[0]+poly1*(phi.T*np.linalg.inv(Phi_1.astype(float))*phi)+poly2/sqrt(k)).item())
            phi=np.matrix([[0],[1],[0],[1]])
            Q[0][3]=min(6,(w1.T*phi+R1[1]+poly1*(phi.T*np.linalg.inv(Phi_1.astype(float))*phi)+poly2/sqrt(k)).item())
            # print("Q",Q)
    # print(revenue)
    # print(benchmark)
    # print(len(revenue))
    # print(len(benchmark))
    plt.plot(revenue,label="CLUB")
    plt.plot(benchmark,label="Benchmark")
    plt.legend()
    # plt.show()
    plt.savefig("./CLUB.png")
    plt.close()
    regret=np.array(benchmark)-np.array(revenue)
    plt.plot(regret,label="Regret")
    plt.legend()
    plt.savefig("./CLUB_regret.png")
    plt.close()
    # print("Q",Q)
    return regret
 
if __name__ == '__main__':
    for i in range(10):      
        K,H=1000,2          
        context=Gen_Context(K) 
        poly1=H*log(K)**2
        poly2=H**2*log(K)**4
        regret=CLUB(K,H,np.matrix([[0.4],[0.6]]),0.9,context,poly1,poly2)
        print("reg",regret[2*K-1])              
                
                    
                
            
            

            
           
            
            

