import numpy as np

def plot_2d_classification(classifier, X, fill=False, ax=None, eps=None, alpha=1, cmap=None, res=1000):                                       
    # multiclass                                                                                                               
    if eps is None:                                                                                                             
      eps = X.std(axis=0) / 2.                                                                                                       
    if ax is None:                                                                                                             
      ax = plt.gca()                                                                                                           
    x_min, x_max = X[:, 0].min() - eps[0], X[:, 0].max() + eps[0]                                                                     
    y_min, y_max = X[:, 1].min() - eps[1], X[:, 1].max() + eps[1]                                                                     
    xx = np.linspace(x_min, x_max, res)                                                                                       
    yy = np.linspace(y_min, y_max, res)                                                                                       
    X1, X2 = np.meshgrid(xx, yy)                                                                                               
    X_grid = np.c_[X1.ravel(), X2.ravel()]                                                                                     
    decision_values = classifier.predict(X_grid)                                                                               
    ax.imshow(decision_values.reshape(X1.shape), extent=(x_min, x_max, y_min, y_max),                                           
              aspect='auto', origin='lower', alpha=alpha, cmap=cmap)                                                               
    ax.set_xlim(x_min, x_max)                                                                                                   
    ax.set_ylim(y_min, y_max)                                                                                                   
    ax.set_xticks(())                                                                                                           
    ax.set_yticks(())  