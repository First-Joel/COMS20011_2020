import os
import sys
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)


def draw_line(X,Wh,func):
    xs = np.linspace(X.min(),X.max(),50)
    plt.plot(xs,func(xs).dot(Wh),lw = 3)
    return None


def linear(X):
    ones = np.ones(X.shape)
    Xe = np.column_stack((ones, X))
    return Xe

def poly(X):
    ones = np.ones(X.shape)
    X2 = X**2
    X3 = X**3
    X4 = X**4
    #X5 = X**5
    #X6 = X**6
    #X7 = X**7
    #X8 = X**8
    #X9 = X**9
    Xe = np.column_stack((ones,X,X2,X3,X4))
    return Xe

def unknown(X):
    ones = np.ones(X.shape)
    XExp = np.exp(X)
    Xe = np.column_stack((ones,X,XExp))
    return Xe

def reconstruction_error(X,Y,Wh,func):
    Yh = func(X).dot(Wh)
    squared_error_vector = (Y-Yh)**2
    squared_error = np.sum(squared_error_vector)
    return squared_error

def determine_function_type(X,Y):
    cross_error_linear = cross_validation(X,Y,linear)
    cross_error_poly = cross_validation(X,Y,poly)
    cross_error_unknown = cross_validation(X,Y,unknown)
    minimum_error = min(cross_error_linear,cross_error_poly,cross_error_unknown)

    if minimum_error == cross_error_linear: 
        print("Chose LINEAR")
        return linear
    if minimum_error == cross_error_poly:
        print("Chose POLY")
        return poly
    if minimum_error == cross_error_unknown:
        print("Chose UNKNOWN")
        return unknown

    return None

def cross_validation(X,Y,func):
    
    indeces_to_remove = np.random.randint(20,size = 6)
    Xminus = np.delete(X,indeces_to_remove,0)
    Yminus = np.delete(Y,indeces_to_remove,0)
    Wh = least_squares(func(Xminus),Yminus)
    cross_error = reconstruction_error(X,Y,Wh,func)
    print("CROSSVALIDATION ERR:",cross_error)
    return cross_error

def least_squares(X, Y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

#Begin
xs, ys = load_points_from_file("datafiles/train_data/basic_2.csv")
view_data_segments(xs,ys)

total_reconstruction_error = 0

for s in range (0,len(xs),20):
    x_seg = xs[s:s+20]
    y_seg = ys[s:s+20]


    WhLin = least_squares(linear(x_seg),y_seg)
    WhPoly = least_squares(poly(x_seg),y_seg)
    WhUnknown = least_squares(unknown(x_seg),y_seg)

    func = determine_function_type(x_seg,y_seg)
    Wh = least_squares(func(x_seg),y_seg)
    error = reconstruction_error(x_seg,y_seg,Wh,func)
    total_reconstruction_error += error

    linearError = reconstruction_error(x_seg,y_seg,WhLin,linear)
    polyError = reconstruction_error(x_seg,y_seg,WhPoly,poly)
    
    print("THIS IS THE ERROR",error)

    #draw_line(xs[s:s+40],Wh,func)
    draw_line(xs[s:s+20],WhPoly,poly)

print(total_reconstruction_error)
plt.show()
#End