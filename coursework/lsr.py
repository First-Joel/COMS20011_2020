import os
import sys
import math
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
    xs = np.linspace(X.min(),X.max(),200)
    plt.plot(xs,func(xs).dot(Wh),lw = 3)
    return None


def linear(X):
    ones = np.ones(X.shape)
    Xe = np.column_stack((ones, X))
    return Xe

def poly(X):
    # Only include the parameters you want in Xe
    ones = np.ones(X.shape)
    X2 = X**2
    X3 = X**3
    Xe = np.column_stack((ones,X,X2,X3))
    return Xe

def poly2(X):
    # Only include the parameters you want in Xe
    ones = np.ones(X.shape)
    X2 = X**2
    Xe = np.column_stack((ones,X,X2))
    return Xe
def poly4(X):
    # Only include the parameters you want in Xe
    ones = np.ones(X.shape)
    X2 = X**2
    X3 = X**3
    X4 = X**4
    Xe = np.column_stack((ones,X,X2,X3,X4))
    return Xe
def poly6(X):
    # Only include the parameters you want in Xe
    ones = np.ones(X.shape)
    X2 = X**2
    X3 = X**3
    X4 = X**4
    X5 = X**5
    X6 = X**6
    Xe = np.column_stack((ones,X,X2,X3,X4,X5,X6))
    return Xe


def unknown(X):
    ones = np.ones(X.shape)
    XSin = np.sin(X)
    Xe = np.column_stack((ones,XSin))
    return Xe

def reconstruction_error(X,Y,Wh,func):
    Yh = func(X).dot(Wh)
    squared_error_vector = (Y-Yh)**2
    squared_error = np.sum(squared_error_vector)
    return squared_error

def determine_function_type(X,Y):
    cross_error_linear = k_fold_cross_validation(X,Y,linear,k)
    cross_error_poly = k_fold_cross_validation(X,Y,poly,k)
    cross_error_unknown = k_fold_cross_validation(X,Y,unknown,k)
    minimum_error = min(cross_error_linear,cross_error_poly,cross_error_unknown)

    if minimum_error == cross_error_linear: 
        #print("Chose LINEAR")
        choices.append("LINEAR")
        return linear
    if minimum_error == cross_error_poly:
        #print("Chose POLY")
        choices.append("POLY")
        return poly
    if minimum_error == cross_error_unknown:
        #print("Chose UNKNOWN")
        choices.append("UNKNOWN")
        return unknown

    return

def determine_poly_order(X,Y):
    cross_error_poly3 = k_fold_cross_validation(X,Y,poly,k)
    cross_error_poly2 = k_fold_cross_validation(X,Y,poly2,k)
    cross_error_poly4 = k_fold_cross_validation(X,Y,poly4,k)
    cross_error_poly6 = k_fold_cross_validation(X,Y,poly6,k)
    minimum_error = min(cross_error_poly2,cross_error_poly3,cross_error_poly4,cross_error_poly6)

    if minimum_error == cross_error_poly2: 
        print("Chose order of 2")
        choices.append("2")
        return poly2
    if minimum_error == cross_error_poly3:
        print("Chose order of 3")
        choices.append("3")
        return poly
    if minimum_error == cross_error_poly4:
        print("Chose order of 4")
        choices.append("4")
        return poly4
    if minimum_error == cross_error_poly6:
        print("Chose order of 6")
        choices.append("6")
        return poly6


    return  

def k_fold_cross_validation(X,Y,func,k):
    k_fold_error = 0
    test_size = math.floor(len(X)/k)
    for s in range (0,len(X)+1-test_size,test_size):
        test_data = np.arange(s,s+test_size,1)
        k_fold_error = k_fold_error + cross_validation(X,Y,func,test_data)
    average_k_fold_error = k_fold_error / k

    #print("K FOLD CROSSVALIDATION ERR ",func.__name__,average_k_fold_error)

    return average_k_fold_error


def cross_validation(X,Y,func,test_data_indices):
    
    #test_data_indices = np.random.randint(20,size = 6)
    Xtraining = np.delete(X,test_data_indices,0)
    Ytraining = np.delete(Y,test_data_indices,0)
    Wh = least_squares(func(Xtraining),Ytraining)
    Xtest = X[test_data_indices]
    Ytest = Y[test_data_indices]
    cross_error = reconstruction_error(Xtest,Ytest,Wh,func)

    return cross_error

def least_squares(X, Y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

#################################
###########   Begin  ############
#################################

k = 4
plot = False
choices = []
if(len(sys.argv)==3):
    if(sys.argv[2] == "--plot"):
        plot = True
if(len(sys.argv) > 1):
    filename = sys.argv[1]
else:
    filename = "basic_5.csv"
xs, ys = load_points_from_file("datafiles/train_data/"+filename)
if(plot):
    view_data_segments(xs,ys)

total_reconstruction_error = 0

for s in range (0,len(xs),20):
    x_seg = xs[s:s+20]
    y_seg = ys[s:s+20]


    WhLin = least_squares(linear(x_seg),y_seg)
    WhPoly = least_squares(poly(x_seg),y_seg)
    WhUnknown = least_squares(unknown(x_seg),y_seg)

    func = determine_function_type(x_seg,y_seg)
    #func = determine_poly_order(x_seg,y_seg)
    Wh = least_squares(func(x_seg),y_seg)
    error = reconstruction_error(x_seg,y_seg,Wh,func)
    total_reconstruction_error += error

    linearError = reconstruction_error(x_seg,y_seg,WhLin,linear)
    polyError = reconstruction_error(x_seg,y_seg,WhPoly,poly)
    
    #print("THIS IS THE ERROR",error)
    
    if(plot):
        draw_line(xs[s:s+20],Wh,func)
        #draw_line(xs[s:s+20],WhLin,linear)
        #draw_line(xs[s:s+20],WhPoly,poly)
        #draw_line(xs[s:s+20],WhUnknown,unknown)
        
        # if(s >= 20):
        #     draw_line(xs[s-20:s+40],WhUnknown,unknown)
        # else:
        #     draw_line(xs[s:s+40],WhUnknown,unknown)

   
print(total_reconstruction_error)
plt.show()
#End