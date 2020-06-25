import cPickle as pickle
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import math
import matplotlib
import scipy.odr as odr

def odr_line(B,x):
    y=B[0]*x+B[1]
    return y

def perform_odr(x,y,xerr,yerr):
    line=odr.Model(odr_line)
    mydata=odr.RealData(x,y,sx=xerr,sy=yerr)
    myodr=odr.ODR(mydata,line,beta0=[1.,0.])
    myoutput=myodr.run()
    myoutput.pprint()
    return myoutput

def subtract(filename1,filename2):
    perturbed=pickle.load(open(filename1,'r'))
    control=pickle.load(open(filename2,'r'))
    x_values=[]
    x_errors=[]
    y_values=[]
    y_errors=[]
    for key in perturbed:
      if key[1] not in ('*','WT'):
	try:
	  if int(control[key][2]) > 1 and int(perturbed[key][2]) > 1:
	    if perturbed[key][1] == 0.:
		print key
	    x_values.append(control[key][0])
	    x_errors.append(control[key][1])
	    y_values.append(perturbed[key][0])
	    y_errors.append(perturbed[key][1])
	except KeyError:
	    donothing=0

    plt.scatter(x_values,y_values)
    plt.show()

    predicted={}
    residuals={}

    regression=perform_odr(x_values,y_values,x_errors,y_errors)
    print regression.beta,regression.sd_beta

    for key in perturbed:
      if key[1] not in ('*','WT'):
        try:
	    predicted[key]=[control[key][0]*regression.beta[0]+regression.beta[1]]
	    error1=predicted[key][0]*np.sqrt((control[key][1]/control[key][0])**2+(regression.sd_beta[0]/regression.beta[0])**2)
	    error=np.sqrt(error1**2+regression.sd_beta[1]**2)
	    predicted[key].append(error)
	    predicted[key].append(control[key][2])

	    residuals[key]=[perturbed[key][0]-predicted[key][0]]
	    error=np.sqrt(perturbed[key][1]**2+predicted[key][1]**2)
	    residuals[key].append(error)
	    residuals[key].append(perturbed[key][2])

	    m1=perturbed[key][0]
	    s1=residuals[key][1]
	    n1=residuals[key][2]
	    m2=predicted[key][0]
	    s2=predicted[key][1]
	    n2=predicted[key][2]

            try:
                p_value=stats.ttest_ind_from_stats(m1,s1,n1,m2,s2,n2,equal_var=False)[1]
            except ZeroDivisionError:
		print "Error!"
                p_value=np.nan
            residuals[key].append(p_value)
        except KeyError:
            donothing=0

    with open(filename1[:-4]+'_residuals.pkl','w') as fout:
        fout.write(pickle.dumps(residuals))
    with open(filename1[:-4]+'_residuals.txt','w') as fout2: # and as text
        for variant in residuals:
            fout2.write(str(variant)+','+str(residuals[variant])+'\n')

    x_values=[]
    y_values=[]
    labels=[]
    for key in residuals:
      try:
	x_values.append(predicted[key][0])
	y_values.append(residuals[key][0])
	labels.append(key)
      except KeyError:
	donothing=0
    plt.scatter(x_values,y_values)
    plt.show()


