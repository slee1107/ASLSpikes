import numpy as np

#confusion matrix a[expected][actual]
confusion = np.zeroes((26, 26))


# This goes in the plotting loop
# test_label = [0, 0, 0, 1.....] x26
expected = np.argmax(test_label)

output_data = np.sum(sim.data[0][i], axis=1) # output for the specific image we're testing 
actual = np.argmax(output_data)
confusion[expected][actual] = confusion[expected][actual] + 1
