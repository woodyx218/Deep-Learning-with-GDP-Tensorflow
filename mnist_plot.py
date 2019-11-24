import pickle
import os
import numpy as np
import pandas as pd
from matplotlib.pyplot import *
from scipy.stats import norm
from scipy import optimize
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from privacy_accountants import *

rc('xtick',labelsize=12)
rc('ytick',labelsize=12)
####### MNIST accuracy boost by adding necessary noise 1
MAnoise=pickle.load(open(os.getcwd()+"/pickle/MNIST_MA1.pkl","rb"))
CLTnoise=pickle.load(open(os.getcwd()+"/pickle/MNIST_CLT1.pkl","rb"))
NOnoise=pickle.load(open(os.getcwd()+"/pickle/MNIST_NO1.pkl","rb"))


l1=plot(np.arange(1,71),MAnoise,linewidth=2,color='royalblue',
                label='MA $\sigma$',linestyle='dashed')
xlabel('epochs',fontsize=15)
ylabel('accuracy',fontsize=15)
title('Accuracy with different noise scales',fontsize=16)
l2=plot(np.arange(1,71),NOnoise,linewidth=2,linestyle='-.',
                   color='k',label='Non-private')
l3=plot(np.arange(1,71),CLTnoise,linewidth=2,
                   color='r',label='CLT $\widetilde\sigma$')
legend(loc='lower right',fontsize=12)
savefig(fname="acc_best.pdf",format="pdf")
show()


#### make sure final epsilon budget is the same
final_eps=compute_epsilon(70,0.7,60000,256,1e-5)
final_geps=compute_epsP(70,0.64,60000,256,1e-5)

eps=[compute_epsilon(i,0.7,60000,256,1e-5) for i in np.arange(1,71)]
geps=[compute_epsP(i,0.64,60000,256,1e-5) for i in np.arange(1,71)]
xlabel('epochs',fontsize=15)
ylabel('$\epsilon$',fontsize=15)
title('Privacy cost versus epochs',fontsize=16)
l1=plot(np.arange(1,71),geps,label='CLT $\epsilon$',linewidth=2,color='red')
l2=plot(np.arange(1,71),eps,label='MA $\epsilon$',linewidth=2,color='royalblue',linestyle='dashed')
legend(loc='lower right',fontsize=12)
savefig(fname="private_everywhere.pdf",format="pdf")
show()



####### MNIST accuracy boost by adding necessary noise 2
MAnoise=pickle.load(open(os.getcwd()+"/pickle/boost2.pkl","rb"))
CLTnoise=pickle.load(open(os.getcwd()+"/pickle/boost1.pkl","rb"))
NOnoise=pickle.load(open(os.getcwd()+"/pickle/boost_nonprivate.pkl","rb"))


l1=plot(np.arange(1,20),MAnoise,linewidth=2,color='royalblue',
                label='MA $\sigma$',linestyle='dashed')
xlabel('epochs',fontsize=15)
ylabel('accuracy',fontsize=15)
title('Accuracy with different noise scales',fontsize=16)
l2=plot(np.arange(1,20),NOnoise,linewidth=2,linestyle='-.',
                   color='k',label='Non-private')
l3=plot(np.arange(1,20),CLTnoise,linewidth=2,
                   color='r',label='CLT $\widetilde\sigma$')
legend(loc='lower right',fontsize=12)
savefig(fname="acc_best2.pdf",format="pdf")
show()


#### make sure final epsilon budget is the same
final_eps=compute_epsilon(20,1.3,60000,256,1e-5)
final_geps=compute_epsP(20,1.06,60000,256,1e-5)

eps=[compute_epsilon(i,1.3,60000,256,1e-5) for i in np.arange(1,21)]
geps=[compute_epsP(i,1.06,60000,256,1e-5) for i in np.arange(1,21)]
xlabel('epochs',fontsize=15)
ylabel('$\epsilon$',fontsize=15)
title('Privacy cost versus epochs',fontsize=16)
l1=plot(np.arange(1,21),geps,label='CLT $\epsilon$',linewidth=2,color='red')
l2=plot(np.arange(1,21),eps,label='MA $\epsilon$',linewidth=2,color='royalblue',linestyle='dashed')
legend(loc='lower right',fontsize=12)
savefig(fname="private_everywhere2.pdf",format="pdf")
show()









###### MNIST epsilon comparision
gdp_epsilon=[compute_epsP(i,0.7,60000,256,1e-5) for i in range(10,101)]
dp_epsilon=[compute_epsilon(i,0.7,60000,256,1e-5) for i in range(10,101)]
l1=plot(np.arange(10,101)*60000/256/1000,gdp_epsilon,label='CLT $\epsilon$',linewidth=2,color='red')
l2=plot(np.arange(10,101)*60000/256/1000,dp_epsilon,label='MA $\epsilon$',linewidth=2,color='royalblue',linestyle='dashed')
xlabel('iterations/1000',fontsize=15)
title('Privacy cost versus iterations',fontsize=16)
autoscale(axis='x',tight=True)
legend(loc='lower right',fontsize=12)
savefig(fname="epsilon_07.pdf",format="pdf")
show()

###### MNIST delta comparision
delta_MA=np.concatenate((np.arange(1e-5,1e-2,1e-4),np.arange(1e-2,1,1e-2)))
dp_epsilon2=[compute_epsilon(100,0.7,60000,256,i) for i in delta_MA]
gdp_mu=compute_muP(100,0.7,60000,256)
delta_CLT=[delta_eps_mu(i,gdp_mu) for i in dp_epsilon2]
l1=plot(dp_epsilon2,delta_CLT,label='CLT $\delta$',linewidth=2,color='red')
l2=plot(dp_epsilon2,delta_MA,label='MA $\delta$',linewidth=2,color='royalblue',linestyle='dashed')
xlabel('$\epsilon$',fontsize=15)
title('$\delta$ versus $\epsilon$',fontsize=16)
legend(loc='upper right',fontsize=12)
savefig(fname="delta_07.pdf",format="pdf")
show()


###### MNIST delta comparision2
delta=np.concatenate((np.arange(1e-6,1.5e-5,1e-6),np.arange(2e-5,1e-4,1e-5)))
dp_epsilon3=[compute_epsilon(100,0.7,60000,256,i) for i in delta]
gdp_epsilon=[compute_epsP(100,0.7,60000,256,i) for i in delta]
l1=plot(delta,gdp_epsilon,label='CLT $\epsilon$',linewidth=2,color='red')
l2=plot(delta,dp_epsilon3,label='MA $\epsilon$',linewidth=2,color='royalblue',linestyle='dashed')
xlabel('$\delta$',fontsize=15)
title('$\epsilon$ versus $\delta$',fontsize=16)
legend(loc='upper right',fontsize=12)
savefig(fname="delta_x.pdf",format="pdf")
show()


###### MNIST noise comparision
dp_epsilon4=[compute_epsilon(100,i,60000,256,1e-5) for i in np.arange(0.7,3,0.1)]
gdp_noise=[noise_multi_from_epsP(i,100,60000,256,1e-5) for i in dp_epsilon4]
l1=plot(dp_epsilon4,gdp_noise,label=r'CLT $\widetilde{\sigma}$',linewidth=2,color='red')
l2=plot(dp_epsilon4,np.arange(0.7,3,0.1),linewidth=2,
        label='MA $\sigma$',color='royalblue',linestyle='dashed')
xlabel('$\epsilon$',fontsize=15)
title('Noise scale versus $\epsilon$',fontsize=16)
autoscale(axis='x',tight=True)
legend(loc='upper right',fontsize=12)
savefig(fname="sigma_best.pdf",format="pdf")
show()

####### MNIST trade-off diagrams
def plot_tradeoff(eps,mu,title_name,save):
    l1=plot(np.arange(0,1.01,0.01),norm.cdf(norm.ppf(1-np.arange(0,1.01,0.01))-mu),color='r',linewidth=2,label=str(mu)+'-GDP by CLT')
    x_array1=np.arange(0,(1-np.exp(-eps))/(np.exp(eps)-np.exp(-eps)),0.01)
    x_array2=np.arange((1-np.exp(-eps))/(np.exp(eps)-np.exp(-eps)),1.01,0.01)
    y_array1=-np.exp(eps)*x_array1+1
    y_array2=-np.exp(-eps)*(x_array2-1)
    l2=plot(np.concatenate((x_array1+0.003,x_array2+0.003)),np.concatenate((y_array1,y_array2)),color='royalblue',linewidth=2,linestyle='--',label='('+str(eps)+',1e-5)-DP by MA')
    xlabel('Type I error',fontsize=15)
    ylabel('Type II error',fontsize=15)
    xlim(0,1);ylim(0,1)
    title(title_name,fontsize=16)
    legend(loc='upper right',fontsize=12)
    gca().set_aspect('equal', adjustable='box')
    savefig(fname=save,format="pdf",bbox_inches='tight')
    show()
    return None

plot_tradeoff(1.19,0.23,'95.0% accuracy, $\sigma=1.3$','tradeoff023.pdf')
plot_tradeoff(3.01,0.57,'96.6% accuracy, $\sigma=1.1$','tradeoff057.pdf')
plot_tradeoff(7.10,1.13,'97.0% accuracy, $\sigma=0.7$','tradeoff113.pdf')
plot_tradeoff(13.27,2.00,'97.6% accuracy, $\sigma=0.6$','tradeoff200.pdf')
plot_tradeoff(18.72,2.76,'97.8% accuracy, $\sigma=0.55$','tradeoff276.pdf')
plot_tradeoff(32.40,4.78,'98.0% accuracy, $\sigma=0.5$','tradeoff478.pdf')




####### MNIST trade-off diagrams
def plot_tradeoff_envelope(mu,sigma,E,title_name,save):
    l1=plot(np.arange(0,1.01,0.01),norm.cdf(norm.ppf(1-np.arange(0,1.01,0.01))-mu),color='r',linewidth=2,label=str(mu)+'-GDP by CLT')
    rowIndex=np.arange(1e-5,0.1,1e-3)
    frame=pd.DataFrame(0,index=rowIndex,columns=np.arange(0,1.01,0.01))
    
    row=0
    for delta in rowIndex:
        eps=compute_epsilon(E,sigma,60000,256,delta)
        breaking=(1-np.exp(-eps))/(np.exp(eps)-np.exp(-eps))*(1-delta)
        x_array1=np.arange(0,breaking,0.01)
        x_array2=np.arange(breaking,1,0.01)
        y_array1=-np.exp(eps)*x_array1+1-delta
        y_array2=np.exp(-eps)*(1-delta-x_array2)
        frame.iloc[row,:len(y_array1)]=y_array1;frame.iloc[row,len(y_array1):]=y_array2; row+=1
    
    l2=plot(np.concatenate((x_array1,x_array2)),frame.max(),color='royalblue',linewidth=2,linestyle='--',label='($\epsilon,\delta$)-DP by MA')
    xlabel('Type I error',fontsize=15)
    ylabel('Type II error',fontsize=15)
    xlim(0,1);ylim(0,1)
    title(title_name,fontsize=16)
    legend(loc='upper right',fontsize=12)
    gca().set_aspect('equal', adjustable='box')
    savefig(fname=save,format="pdf",bbox_inches='tight')
    show()
    return None

plot_tradeoff_envelope(0.23,1.3,15,'95.0% accuracy, $\sigma=1.3$','envelope_tradeoff023.pdf')
plot_tradeoff_envelope(0.57,1.1,60,'96.6% accuracy, $\sigma=1.1$','envelope_tradeoff057.pdf')
plot_tradeoff_envelope(1.13,0.7,45,'97.0% accuracy, $\sigma=0.7$','envelope_tradeoff113.pdf')






####### MNIST trade-off diagrams, with shade
def plot_tradeoff_envelopeS(mu,sigma,E,title_name,save):
    l1=plot(np.arange(0,1.01,0.01),norm.cdf(norm.ppf(1-np.arange(0,1.01,0.01))-mu),color='r',linewidth=2,label=str(mu)+'-GDP by CLT')
    
    row=0
    for delta in np.arange(1e-5,1,1e-3):
        eps=compute_epsilon(E,sigma,60000,256,delta)
        breaking=(1-np.exp(-eps))/(np.exp(eps)-np.exp(-eps))*(1-delta)
        x_array1=np.arange(0,breaking,0.01)
        x_array2=np.arange(breaking,1,0.01)
        y_array1=-np.exp(eps)*x_array1+1-delta
        y_array2=np.exp(-eps)*(1-delta-x_array2)
        plot(np.concatenate((x_array1+0.002,x_array2+0.002)),np.concatenate((y_array1,y_array2)),color='royalblue',linewidth=2)
    
    l2=plot(np.concatenate((x_array1+0.002,x_array2+0.002)),np.concatenate((y_array1,y_array2)),color='royalblue',linewidth=2,label='($\epsilon,\delta$)-DP by MA')
    xlabel('Type I error',fontsize=15)
    ylabel('Type II error',fontsize=15)
    xlim(0,1);ylim(0,1)
    title(title_name,fontsize=16)
    legend(loc='upper right',fontsize=12)
    gca().set_aspect('equal', adjustable='box')
    savefig(fname=save,format="pdf",bbox_inches='tight')
    show()
    return None

plot_tradeoff_envelopeS(0.23,1.3,15,'95.0% accuracy, $\sigma=1.3$','envelope_tradeoff023S.pdf')
plot_tradeoff_envelopeS(0.57,1.1,60,'96.6% accuracy, $\sigma=1.1$','envelope_tradeoff057S.pdf')
plot_tradeoff_envelopeS(1.13,0.7,45,'97.0% accuracy, $\sigma=0.7$','envelope_tradeoff113S.pdf')
