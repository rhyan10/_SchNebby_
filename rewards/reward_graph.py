import matplotlib.pyplot as plt
import numpy as np

with open('rewardsum.txt', 'r') as f:
    # Read the lines of the file into a list
    lines = f.readlines()

# Convert each line to a float and store in a new list
float_list = [float(line.strip()) for line in lines][:5000]

# Define the window size for the moving average
window_size1 = 100
window_size2 = 200
window_size3 = 300

# Define a function to calculate the moving average
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Calculate the moving average of float_list
ma = moving_average(float_list, window_size1)
ma2 = moving_average(float_list, window_size2)
ma3 = moving_average(float_list, window_size3)

plt.plot(ma)
plt.plot(ma2)
plt.plot(ma3)

# Add labels and title to the graph
plt.xlabel('Step')
plt.ylabel('Reward (1/Fmax)')

# Display the graph
plt.show()
print(1)