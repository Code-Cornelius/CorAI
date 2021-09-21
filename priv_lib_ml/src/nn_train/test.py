# import seaborn as sns
# from priv_lib_plot import APlot
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Load the example diamonds dataset
# diamonds = sns.load_dataset("diamonds")
# print(diamonds)
# # Draw a scatter plot while assigning point colors and sizes to different
# # variables in the dataset
# clarity_ranking = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
# cm = plt.get_cmap("inferno")
# sns.scatterplot(x="carat", y="price",
#                 hue="clarity", size="color",
#                 palette='PuOr_r',
#                 sizes=(1, 8), linewidth=0,
#                 data=diamonds)
#
#
# plt.figure()
# # Load an example dataset with long-form data
# fmri = sns.load_dataset("fmri")
# print(fmri)
# # Plot the responses for different events and regions
# sns.lineplot(x="timepoint", y="signal",
#              hue="region", style="event",
#              palette ='magma',
#              data=fmri)
#
# APlot.show_plot()
#
# #



