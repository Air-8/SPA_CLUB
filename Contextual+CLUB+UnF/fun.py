import numpy as np
import random
from math import sqrt
from scipy.optimize import minimize
import math

def Context():
    a=random.random()
    if (a>0.5):
        x=np.matrix([[0],[1]])
    else:
        x=np.matrix([[1],[0]])
    return x

def Pi_Rand():
    a=3*random.random()
    return a

def CDF(x):
    if (x<-1):
        return 0
    elif (x>1):
        return 1
    else:
        return (x+1)/2

def Pi(x,theta):
    mu=x.T*theta
    return 1+mu.item()/2

def Bid(x,theta,flag,K,gamma):
    scale=3*sqrt(2)/K/sqrt(1-gamma)
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

def UnPi(x,theta_hat,CDF_est):
    mu=(x.T*theta_hat).item()
    reserve=0
    max=0
    for i in range(301):
        y=i/100
        if(y-1-mu<=-1):
            rew=y
        elif(y-1-mu>=1):
            rew=0
        else:
            num=math.floor(100*(y-mu))
            rew=y*(1-CDF_est[num])
        if(rew>max):
            max=rew
            reserve=y
    return reserve
            
        
    
def UnUpdate(outcome,x,k):
    def objective_function(theta):
        sum=0
        for i in range(k-1):
            sum+=(3*outcome[i]-1-x[i][0]*theta[0]-x[i][1]*theta[1])**2
        return sum
    constraint = {'type': 'ineq', 'fun': lambda theta: 2*np.sqrt(2) - np.linalg.norm(theta)}
    initial_guess = [0.0, 0.0]

    result = minimize(objective_function, initial_guess, constraints=constraint)
    est=np.matrix([[result.x[0]],[result.x[1]]])
    # print(est)
    return est

def CDF_Update(bid_record,theta,context,k):
    est=[]
    for i in range(201):
        num=0
        for j in range(k-1):
            thre=bid_record[j]-1-(context[j].T*theta).item()
            # print(thre)
            if(thre<-1):
                thre=-1
            elif(thre>1):
                thre=1
            if(thre<=-1+i/100):
                num+=1
        est.append(num/(k-1))
    # print(est)
    return est