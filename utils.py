import numpy as np

def plot_2d_classification(classifier, X, fill=False, ax=None, eps=None, alpha=1):                                       
    # multiclass                                                                                                               
    if eps is None:                                                                                                             
      eps = X.std() / 2.                                                                                                       
    if ax is None:                                                                                                             
      ax = plt.gca()                                                                                                           
    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps                                                                     
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps                                                                     
    xx = np.linspace(x_min, x_max, 100)                                                                                       
    yy = np.linspace(y_min, y_max, 100)                                                                                       
    X1, X2 = np.meshgrid(xx, yy)                                                                                               
    X_grid = np.c_[X1.ravel(), X2.ravel()]                                                                                     
    decision_values = classifier.predict(X_grid)                                                                               
    ax.imshow(decision_values.reshape(X1.shape), extent=(x_min, x_max, y_min, y_max),                                           
            aspect='auto', origin='lower', alpha=alpha)                                                               
    ax.set_xlim(x_min, x_max)                                                                                                   
    ax.set_ylim(y_min, y_max)                                                                                                   
    ax.set_xticks(())                                                                                                           
    ax.set_yticks(())  