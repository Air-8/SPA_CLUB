import numpy as np
import random
from math import sqrt
from scipy.optimize import minimize

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
    val=1+theta.T*x+2*random.random()-1
    if (flag==0):
        bid=val+scale*(2*random.random()-1)
        if(bid<0):
            bid=0
        elif(bid>3):
            bid=3
    else:
        bid=3*random.random()
    return bid

#k-1 data points  
def Update(outcome,x,reserve,k):
    def objective_function(theta):
        sum=0
        for i in range(k-1):
            sum+=(outcome[i]-1+CDF(reserve[i]-1-x[i][0]*theta[0]-x[i][1]*theta[1]))**2
        return sum
    constraint = {'type': 'ineq', 'fun': lambda theta: 2*np.sqrt(2) - np.linalg.norm(theta)}
    initial_guess = [0.0, 0.0]

    result = minimize(objective_function, initial_guess, constraints=constraint)
    est=np.matrix([[result.x[0]],[result.x[1]]])
    # if(est[0]**2+est[1]**2>8):
    #     est=est/np.sqrt(est[0]**2+est[1]**2)*2*np.sqrt(8)
    # print(est)
    return est

    
    


    
    
