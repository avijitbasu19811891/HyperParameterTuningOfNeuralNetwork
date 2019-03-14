import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as patches




LABEL_SIZE = 14
VALUE_SIZE = 12
LABEL_ROTATION = 0

"""
 Plot the matrix: show and save as image
"""
def plotMatrix(cm, labels, title, fname):
    # Transform to percentage
    """
    for row in range(0, len(cm)):
        rowSum = np.sum(cm[row])
        cm[row] = cm[row] / rowSum * 100
    """
    print("> Plot confusion matrix to", fname, "...")
    fig = plt.figure()

    # COLORS
    # ======
    # Discrete color map: make a color map of fixed colors
    # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    color_list = []
    for i in range(10):
        i += 1
        color_list.append((i*0.1, i*0.1, i*0.1))
    # Reversed gray scale (black for 100, white for 0)
    color_map = colors.ListedColormap(list(reversed(color_list)))
    # Set color bounds: [0,10,20,30,40,50,60,70,80,90,100]
    bounds=range(0, 110, 10)
    norm = colors.BoundaryNorm(bounds, color_map.N)

    # Plot matrix (convert numpy to list) and set Z max to 100
    plt.imshow(cm, interpolation='nearest', cmap=color_map, vmax=100)
    plt.title(title)
    plt.colorbar()

    # LABELS
    # ======
    # Setup labels (same for both axises)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, fontsize=LABEL_SIZE, rotation=LABEL_ROTATION)
    plt.yticks(tick_marks, labels, fontsize=LABEL_SIZE)
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')

    # Save as an image
    # Higher DPI to avoid mis-alignment
    plt.savefig(fname, dpi=240)
    # Show the plot in a window
    plt.close()

"""
     Plot graph of Host Monitored Params and corresponding class
     This class can be either of predicted class or the labeled class
"""
def PlotData(data,labels, fileInfo, Caption):
    shape = [data.shape[1], data.shape[0]]

    dataToPlot = np.reshape(data, shape)

    dataToPlot = dataToPlot[:3]

    """
    plt.plot(dataToPlot)
    plt.plot(predLabels)
    plt.title("Prediction")
    plt.ylabel('Values')
    plt.xlabel('Time')
    #plt.legend(['Train', 'Test'], loc='upper left')
    """
    fig = plt.figure()
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    ax1.plot(dataToPlot)
    ax2.plot(labels)

    ax1.get_shared_x_axes().join(ax1, ax2)
    ax1.set_xticklabels([])

    ax1.yaxis.set_label("Vm CPU, Mem, I/O load")
    ax1.yaxis.set_label("Class")
    plt.savefig(fileInfo)
    plt.close()