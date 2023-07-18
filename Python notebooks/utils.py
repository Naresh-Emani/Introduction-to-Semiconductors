###################################################################################
#==================== Import basic routines =======================================
###################################################################################

import sys
version=sys.version_info.major
import os
import numpy as np
import scipy as sp
import scipy.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl
import ipywidgets as widgets #Slider, Button, RadioButtons
import scipy.linalg as spla
import warnings
warnings.filterwarnings('ignore')

from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import display
from IPython.display import clear_output

try:
    from colorama import Fore,Back,Style
    from colorama import init
    init(autoreset=True)
    print_color=True
except:
    print_color=False
titles={
    1:'Particle in an infinite potential well',
    2:'Particle in a finite well',
    3:'Particle in a double finite well (equal depth)',
    4:'Particle in a double finite well (unequal depth)',
    5:'Particle in a harmonic well',
    6:'Particle in a Morse well',
    7:'Kronig-Penney finite well'}

######################################################################################
#====================== Interactive notebook functions =============================
######################################################################################

#--------------------------  Inf well function --------------------------------------
def inf_well(W,n):
    
    # atomic units
    hbar=1.0
    m=1.0
    #set precision of numerical approximation
    steps=2000
    # divide by two so a well from -W to W is of input width
    W=W / 2.0
    # create x-vector from -W to W
    xvec=np.linspace(-W,W,steps,dtype=np.float_)

    # get step size
    h=xvec[1]-xvec[0]
    # create Laplacian via 3 point finite-difference method
    Laplacian=(-2.0*np.diag(np.ones(steps))+np.diag(np.ones(steps-1),1)\
        +np.diag(np.ones(steps-1),-1))/(float)(h**2)

    # create Hamiltonian
    Hamiltonian=((-0.5*(hbar**2)/m))*Laplacian

    # diagonalize the Hamiltonian yielding the wavefunctions and energies
    E,V=spla.eigh(Hamiltonian)


    # create plot
    infinite_well_plot(E,V,xvec,W,steps,n)
    
    
#--------------------------------- Finite potential well ---------------------------------------
def fin_well(A,D):
    
    # PARTICLE IN A FINITE WELL OF WIDTH (W) AND DEPTH (D)
    # atomic units
    hbar=1.0
    m=1.0
    #set precision of numerical approximation
    steps=2000
    W=A/2
    # create x-vector from -W to W
    xvec=np.linspace(-A,A,steps,dtype=np.float_)
    # get step size
    h=xvec[1]-xvec[0]
    # create the potential from step function
    U=-D*(step_func(xvec+W)-step_func(xvec-W))
    # create Laplacian via 3-point finite-difference method
    Laplacian=(-2.0*np.diag(np.ones(steps))+np.diag(np.ones(steps-1),1)\
        +np.diag(np.ones(steps-1),-1))/(float)(h**2)
    # create the Hamiltonian
    Hamiltonian=np.zeros((steps,steps))
    [i,j]=np.indices(Hamiltonian.shape)
    Hamiltonian[i==j]=U
    Hamiltonian+=(-0.5)*((hbar**2)/m)*Laplacian
    # diagonalize the Hamiltonian yielding the wavefunctions and energies
    E,V=spla.eigh(Hamiltonian)
    # determine number of energy levels to plot (n)
    n=0
    while E[n]<0:
        n+=1
    # print output
    # create plot
    finite_well_plot(E,V,xvec,steps,n,U)

    
#--------------------------------- Quiz function -----------------------------------------------
def create_multipleChoice_widget(description, options, correct_answer):
    if correct_answer not in options:
        options.append(correct_answer)
    
    correct_answer_index = options.index(correct_answer)
    
    radio_options = [(words, i) for i, words in enumerate(options)]
    alternativ = widgets.RadioButtons(
        options = radio_options,
        description = '',
        disabled = False
    )
    
    description_out = widgets.Output()
    with description_out:
        print(description)
        
    feedback_out = widgets.Output()

    def check_selection(b):
        a = int(alternativ.value)
        if a==correct_answer_index:
            s = '\x1b[6;30;42m' + "Correct" + '\x1b[0m' +"\n" #green color
            
        else:
            s = '\x1b[5;30;41m' + "Incorrect" + '\x1b[0m' +"\n" #red color
        with feedback_out:
            clear_output()
            print(s)
        return
    
    check = widgets.Button(description="submit")
    check.on_click(check_selection)
    
    
    return widgets.VBox([description_out, alternativ, check, feedback_out])   

##########################################################################################################
#============================== Add questions for quiz here ==============================================
########################################################################################################## 
def quiz():
    Q1 = create_multipleChoice_widget('1. The eigen energies of an electron in an infinite well are:',
                                      ['continuous','discrete','Insufficient data'],'discrete')
    Q2 = create_multipleChoice_widget('2. The eigen functions penetrate outside the walls of an infinite potential well',
                                      ['True','False','Insufficient data'],'False')
    Q3 = create_multipleChoice_widget('3. For a well with W=3 and n=3, the eigen energies for an electron in the first, second and third eigenstates respectively, are',
                                      ['4.925, 2.189, 0.547','0.547,2.189,4.925','2.189, 0.547, 4.925'],'0.547,2.189,4.925')
    Q4 = create_multipleChoice_widget('4. Let W=1 and n=3, the probability of finding a electron of second eigenstate exactly in the middle of the well is',
                                      ['lowest','highest','cannot determine'],'lowest')
    Q5 = create_multipleChoice_widget('5. The values of energy difference (E3-E2) for well widths W=1 and W=2 respectively, are',
                                      ['14.775 and 3.694','39.399 and 9.85','24.624 and 6.156'],'24.624 and 6.156')
    Q6 = create_multipleChoice_widget('6. With increasing width of potential well, the difference between two consecutive eigen energies',
                                      ['decreases linearly','increases quadratically','decreases quadratically'],'decreases quadratically')
    
    display(Q1);display(Q2);display(Q3);display(Q4);display(Q5);display(Q6);
   

        
               

        
########################################################################################################
#============================= Basic functions for formatting and plotting =============================
########################################################################################################


def infinite_well_input(W=None,n=None):
    if W==None:
        try:
            W=float(input('\nEnter the width of your infinite well in atomic units (a.u.).\n\tSelect a value between 0.5 and 15: '))
            W,n=infinite_well_input(W=W)
        except ValueError:
            valid_input_error_message()
            W,n=infinite_well_input()
    else:
        try:
            n=int(input('Enter the number of wavefunctions you would like to plot.\n\tThis value must be an integer: '))
        except ValueError:
            valid_input_error_message()
            W,n=infinite_well_input(W=W)
    return W,n


def ask_to_save_plot(error=False):
    if error==True:
        valid_input_error_message()
    try:
        image=input('Would you like to save a .png image of your plot? Type yes or no. ')
    except:
        image=ask_to_save_plot(error=True)
    image=image.strip().lower()
    if image=='yes':
        print('Your image will be saved in your current working directory.')
    if image not in {'yes','no'}:
        image=ask_to_save_plot(error=True)
    return image
def ask_to_plot_squared(error=False):
    if error==True:
        valid_input_error_message()
    try:
        sq=input('Would you like to plot the probability density (psi squared) instead of the probability amplitude (psi)? Type yes or no. ')
    except:
        sq=ask_to_plot_squared(error=True)
    sq=sq.strip().lower()
    if sq not in {'yes','no'}:
        sq=ask_to_plot_squared(error=True)
    return sq
def print_number_of_wavefunctions(n):
    if print_color:
        print(Fore.RED+'\nMaximum number of wavefunctions for plotting is', Fore.RED + str(n), "\n")
    else:
        print('\nMaximum number of wavefunctions for plotting is', n)

################################################################################
# SHARED FUNCTIONS
################################################################################
def step_func(x):
    return 0.5*(1+np.sign(x))
def harmonic_potential(x,omega,D):
    pot=(0.5*(omega**2)*(x**2))+D
    for i in range(len(pot)):
        if pot[i]>0:
            pot[i]=0
    return pot
def morse_function(a,D,x):
    return D*(np.exp(-2*a*x)-2*np.exp(-a*x))
def morse_potential(omega,D,steps):
    D=np.abs(D)
    a=np.sqrt(omega/2.0*D)
    start=0.0
    stop=0.0
    while morse_function(a,D,start)<0.5*np.abs(D):
        start-=0.01
    while morse_function(a,D,stop)<-0.1:
        stop+=0.01
    # create x-vector
    xvec=np.linspace(2.0*start,2.0*stop,steps,dtype=np.float_)
    # get step size
    h=xvec[1]-xvec[0]
    pot=morse_function(a,D,xvec)
    for i in range(len(pot)):
        if pot[i]>0:
            pot[i]=0
    return xvec,h,pot
def diagonalize_hamiltonian(Hamiltonian):
    return spla.eigh(Hamiltonian)

#####################################################################################
# PLOTTING
####################################################################################
def infinite_well_plot(E,V,xvec,W,steps,n,ask_to_save=False,ask_squared=False):
    if ask_squared:
        sq=ask_to_plot_squared()
        if(sq=='yes'):
            V = np.multiply(np.conj(V),V)
    V_new,ScaleFactor=infinite_well_plot_scaling(E,V,xvec,W)
    # create the figure
    f=plt.figure()
    # add plot to the figure
    ax=f.add_subplot(111)
    # set x limit
    plt.xlim(-W,W)
    # determine how much to buffer the axes
    buff=(np.max(V_new[0:steps,n-1])-np.min(V_new[0:steps,n-1]))
    #set y limit
    plt.ylim(0,np.max(V_new[0:steps,n-1])+buff)
    #plot wave functions
    for i in np.arange(n-1,-1,-1):
        color=mpl.cm.jet_r((i)/(float)(n-1),1)
        wavefunc=ax.plot(xvec,V_new[0:steps,i],c=color,label='E(a.u.)={}'.format(np.round(E[i]*1000)/1000.0))
        ax.axhline(y=V_new[0,i],xmin=-20*W,xmax=20*W,c=color,ls='--')
    # set plot title
    ax.set_title('Infinite potential well')
    # set x label
    plt.xlabel('Width of Well / (a.u.)')
    # set y label
    plt.ylabel('Energy / (a.u.)')
    # modify tick marks
    ax.set_yticklabels(np.round(ax.yaxis.get_ticklocs()*ScaleFactor))
    # add plot legend
    L=plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
    box=ax.get_position()
    ax.set_position([box.x0,box.y0,0.7*box.width,box.height])
    plt.show()


def finite_well_plot(E,V,xvec,steps,n,U,ask_to_save=False,ask_squared=False):
    if ask_squared:
        sq=ask_to_plot_squared()
        if(sq=='yes'):
            V = np.multiply(np.conj(V),V)
    V_new,ScaleFactor,U_new,n=finite_well_plot_scaling(E,V,xvec,U,n,steps)
    # create the figure
    f=plt.figure()
    # add plot to the figure
    ax=f.add_subplot(111)
    # plot potential
    ax.plot(xvec,U_new,c='lightslategray')
    # find appropriate x limits and set x limit
    MinX=0
    MaxX=len(xvec)-1
    while U_new[MinX]==0:
        MinX=MinX+1
    while U_new[MaxX]==0:
        MaxX=MaxX-1
    for m in range(n):
        V_old=V_new[MinX+1,m]
        while(np.abs(V_old - V_new[MinX,m])>1e-6 and MinX>0):
            V_old=V_new[MinX,m]
            MinX=MinX-1
        V_old=V_new[MaxX-1,m]
        while(np.abs(V_old - V_new[MaxX,m])>1e-6 and MaxX<len(xvec)-1):
            V_old=V_new[MaxX,m]
            MaxX=MaxX+1
    plt.xlim(xvec[MinX],xvec[MaxX])
    # find appropriate y limits and set y limit
    if(np.max(V_new)>0):
        if(np.min(V_new)>np.min(U_new)):
            plt.ylim(1.05*np.min(U_new),np.max(V_new)+abs(0.05*np.min(U_new)))
        else:
            plt.ylim(1.05*np.min(V_new),np.max(V_new)+abs(0.05*np.min(U_new)))
    else:
        if(np.min(V_new)>np.min(U_new)):
            plt.ylim(1.05*np.min(U_new),np.max(U_new)+abs(0.05*np.min(U_new)))
        else:
            plt.ylim(1.05*np.min(V_new),np.max(U_new)+abs(0.05*np.min(U_new)))
    #plot wave functions
    for i in np.arange(n-1,-1,-1):
        color=mpl.cm.jet_r((i)/(float)(n),1)
        wavefunc=ax.plot(xvec,V_new[0:steps,i],c=color,label='E(a.u.)={}'.format(np.round(E[i]*1000)/1000.0))
        ax.axhline(y=V_new[0,i],xmin=-10,xmax=10,c=color,ls='--')
    # set plot title
    ax.set_title('Finite potential well')
    # set x label
    plt.xlabel('Width of Well / (a.u.)')
    # set y label
    plt.ylabel('Energy / (a.u.)')
    # modify tick marks
    ax.set_yticklabels(np.round(ax.yaxis.get_ticklocs()*ScaleFactor))
    # add plot legend
    L=plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
    box=ax.get_position()
    ax.set_position([box.x0,box.y0,0.7*box.width,box.height])
    
    plt.show()

    
def infinite_well_plot_scaling(E,V,xvec,W):
    # scale the wave functions
    ScaleFactorStep=0.05
    ScaleFactor=1.00
    MaxV2=np.amax(V[1])
    MinV2=np.amin(V[1])
    MaxV1=np.amax(V[0])
    while((MaxV2-MinV2)<np.abs(MinV2-MaxV1)*10.0):
        MaxV2=np.amax(V[1])+E[1]/ScaleFactor
        MinV2=np.amin(V[1])+E[1]/ScaleFactor
        MaxV1=np.amax(V[0])+E[0]/ScaleFactor
        ScaleFactor+=ScaleFactorStep
    V_new=(E/ScaleFactor)+V
    return V_new,ScaleFactor


def finite_well_plot_scaling(E,V,xvec,U,n,steps):
    # scale the wave functions
    order=np.argsort(E)
    Converged=False
    while(Converged is False):
        E_copy=E[0:n]
        V_copy=V[0:steps,order]
        V_copy=V[0:steps,0:n]
        max_E_diff = E_copy[n-1] - E_copy[0]
        found_step = False
        step = 1
        while found_step is False:
            if(E_copy[step]-E_copy[0]<0.2):
                step+=1
            else:
                found_step = True
        ScaleFactorStep=0.05
        ScaleFactor=1.00
        Overlap=1
        while(Overlap==1):
            for i in range(0,n,step):
                MaxV2=np.max(V_copy[0:steps,i])+E_copy[i]/ScaleFactor
                MinV2=np.min(V_copy[0:steps,i])+E_copy[i]/ScaleFactor
                MaxV1=np.max(V_copy[0:steps,i-step])+E_copy[i-step]/ScaleFactor
                if((MaxV2-MinV2)<(np.abs(MinV2-MaxV1)*10)):
                    Overlap=1
                else:
                    Overlap=0
                    break
            ScaleFactor=ScaleFactor+ScaleFactorStep
        V_copy_new=(E_copy/ScaleFactor)+V_copy
        if np.max(V_copy_new[n])>0:
            Converged=True
            n=n-1
        else:
            n=n+1
        V_copy_old=V_copy_new
    V_new=V_copy_old
    U_new=U/ScaleFactor
    return V_new,ScaleFactor,U_new,n