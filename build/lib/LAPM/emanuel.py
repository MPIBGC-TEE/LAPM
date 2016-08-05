"""Example: Emanuel's model"""

from __future__ import division
from sympy import Matrix, exp, symbols, latex
from sympy.printing import pretty, pretty_print, pprint
from LAPM.LinearAutonomousPoolModel import LinearAutonomousPoolModel
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np


def plot_emanuel_ages(EM):
    """Plot system content versus system age."""
    # matplotlib setup
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams['font.family'] = 'Liberation Sans'
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1,1,1)

    # plot for ages from 0 to 100
    xdata = np.arange(0, 100.1, 0.1)
    
    # compute steady state system content
    content = sum(EM.xss)

    # multiply age density with system content to get system's age distribution
    ydata = [EM.A_density(age=x) for x in xdata]
    ydata = [y*content for y in ydata]

    # plot system content versus age
    ax.plot(xdata, ydata, color='black')
    
    # mean system age, system content at mean system age
    mx, my = (EM.A_expected_value, EM.A_density(age=EM.A_expected_value)*content)
    
    # plot vertical line for mean
    line = ax.plot([mx, mx], [0, my], color="black", linestyle='dashed', linewidth=1)
    line[0].set_dashes([1,1])
    
    ax.annotate('total system', xy=(0,0), ha='right', textcoords='axes fraction', xytext=(0.95,0.85), fontsize=20)

    # write mean
    ax.annotate('$\\mu=%1.2f$' % mx, xy=(0,0), ha='right', textcoords='axes fraction', xytext=(0.95,0.75), fontsize=16)

    # write standard deviation
    ax.annotate('$\\sigma=%1.2f$' % EM.A_standard_deviation, xy=(0,0), ha='right', textcoords='axes fraction', xytext=(0.95,0.65), fontsize=16)
    
    # improve plot layout
    ax.set_xlim([0, xdata[-1]])
    ax.set_ylim([0, 120])
    
    ax.set_xticks(np.linspace(0, xdata[-1], 6))
    minor_locator = AutoMinorLocator(5)
    ax.xaxis.set_minor_locator(minor_locator)
    
    ax.set_yticks(np.linspace(0, 120, 5))
    minor_locator = AutoMinorLocator(5)
    ax.yaxis.set_minor_locator(minor_locator)
    
    # set tick fontsize
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # axes labels
    ax.set_xlabel("age in years", fontsize=16)
    ax.set_ylabel("system content in Pg C", fontsize=16)

    plt.show()   


def emanuel():
    """Initialize Emanuel's model, show the functionality.

    Look at the source code!
    """
    A = Matrix([[-2.0810,        0,       0,       0,        0],
                [ 0.8378,  -0.0686,       0,       0,        0],
                [      0,        0, -0.5217,       0,        0],
                [ 0.5676,   0.0322,  0.1739, -0.5926,        0],
                [      0, 4.425e-3,  0.0870, 0.0370, -9.813e-3]])

    u = Matrix(5, 1, [77, 0, 36, 0, 0])

    # EM is now Emanuel's model
    EM = LinearAutonomousPoolModel(u, A) 
    t, y = symbols('t y')

    # outputs transit time density formula
    print('The transit time density.')
    print((EM.T_density()))
    print()

    # outputs transit time density value at t=5
    print('The transit time density at t=5.')
    print(EM.T_density(5))
    print()
    
    # outputs the third moment of the pool age vector
    print('The vector of third moments of the pool ages.')
    print(EM.a_nth_moment(3))
    print()

    # plot system age distribution
    print('A plot of the system age distribution is being created.')
    plot_emanuel_ages(EM)


#################################################################

if __name__ == '__main__':
    emanuel()



