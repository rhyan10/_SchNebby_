import matplotlib.pyplot as plt
import numpy as np
import ase.io
import dill as pickle

size=20
params={'legend.fontsize': 'large',
       'figure.figsize': (7,5),
        'axes.labelsize':size,
        'axes.titlesize':size,
        'ytick.labelsize':size,
        'xtick.labelsize':size,
        'axes.titlepad':10
       }
plt.rcParams.update(params)
fig=plt.figure()


true_energies = np.array([-463.11678, -463.10602, -463.1086, -463.0945, -463.07201, -463.06049, -463.07136, -463.08837, -463.09647, -463.09901])[::2]
true_energies = (true_energies - true_energies[0])*630

schnebby_energies =  np.array([-0.06289946, -0.05993157, -0.04560327, -0.00441635, -0.00784401, -0.00817167, -0.0062098, -0.00375224, -0.02834004, -0.04547545])[::2]
schnebby_energies = (schnebby_energies - schnebby_energies[0])*630

initial_interpolation_energy = np.array([-0.06289946, -0.04735102, -0.005724, 0.047844503, 0.12370, 0.20208, 0.20169821, 0.09456, 0.012956, -0.04547])[::2]
initial_interpolation_energy = (initial_interpolation_energy - initial_interpolation_energy[0])*630

x = [1,3,5,7,9]

# # Define a function to calculate the moving average
# def moving_average(x, w):
#     return np.convolve(x, np.ones(w), 'valid') / w

# # Calculate the moving average of float_list
# ma = moving_average(schnebby_energies, 3) 
# ma = np.concatenate((np.array([0]),ma))

plt.plot(x,true_energies, label="Quantum Chemistry MEP", color="midnightblue",linewidth=5)
plt.plot(x,schnebby_energies, label="SchNebby MEP",color="lightcoral",linewidth=5)
plt.plot(x,initial_interpolation_energy, label="Initial Pathway",color="gold",linewidth=5)
plt.legend(fontsize="x-large",loc="upper left")
# Add labels and title to the graph
plt.xlabel('Image Number')
plt.ylabel('Energy (kcal/mol)')

# Display the graph
plt.show()
print(1)
