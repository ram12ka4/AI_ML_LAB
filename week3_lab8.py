# Ungraded Lab: Overfitting

# Goals

# In this lab, you will explore:

#     the situations where overfitting can occur
#     some of the solutions

import matplotlib.pyplot as plt
from ipywidgets import Output
from IPython.display import display
from plt_overfit import overfit_example, output

plt.style.use("./deeplearning.mplstyle")

plt.close("all")
display(output)
ofit = overfit_example(False)
