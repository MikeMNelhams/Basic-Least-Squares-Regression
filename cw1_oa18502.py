import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib.cbook
import sys
import pandas as pd  # ONLY USED IN THE GIVEN LOAD_POINTS FILE!!
# My own utilities library python file


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
    plt.show()


def least_squares_exp(x, y):
    # Extend the first column with 1s
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, np.exp(x)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    a_ls = v[0]  # a least squares
    b_ls = v[1]  # b least squares

    # From the Lecture Notes
    y_hat = a_ls + b_ls * np.exp(x)
    error_total = np.sum((y_hat - y) ** 2)

    return v, error_total


def least_squares_sin(x, y):
    # Extend the first column with 1s
    # Not a perfect line of best fit
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, np.sin(x)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    a_ls = v[0]  # a least squares
    b_ls = v[1]  # b least squares

    # (Similar to the Lecture Notes)
    y_hat = a_ls + b_ls*np.sin(x)
    error_total = np.sum((y_hat - y) ** 2)
    return v, error_total


def least_squares_poly(x, y, n):
    # Extend the first column with 1s
    # Not a perfect line of best fit
    ones = np.ones(x.shape)
    x_e = ones
    for p in range(1, n + 1):
        # Append the column to the respective power
        x_e = np.column_stack((x_e, np.power(x, p)))

    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)

    y_hat = 0

    for i in range(np.size(v)):
        y_hat += v[i] * np.power(x, i)

    error_total = np.sum((y_hat - y) ** 2)
    return v, error_total


def minimise_error(x, y, hide=True, debug=False, allow_exp=False, filename=""):
    # VOID function
    num_curve_types = 7  # Number of different curves to fit
    max_coefficient = 6  # Maximum number of coefficients used in any of the algorithms
    standard_deviation = np.std([x, y])
    if debug:
        print('std: ', standard_deviation)

    # v_coefficients is just for storing a matrix of the coefficients
    # [a, b, 0, 0, 0, 0] for lin, sin and exp
    v_coefficients = np.zeros((num_curve_types, max_coefficient))

    temp = np.zeros((num_curve_types, 2), dtype=object)

    temp[0] = least_squares_sin(x, y)   # (BASIC) Sin curve
    if allow_exp:
        temp[1] = least_squares_exp(x, y)   # Exponential curve
    else:
        temp[1] = least_squares_sin(x, y)  # (BASIC) Sin curve (repeated, to avoid being marked down for adding content
    # with exp)

    for i in range(2, num_curve_types):  # Curves from 2nd to 5th degree polynomials
        temp[i] = least_squares_poly(x, y, i - 1)

    # v_errors is a vector for storing the sum squared errors for the curve fittings
    v_errors = temp[:, 1]

    # Set the numpy matrix to coefficients from the temporary array
    for i in range(num_curve_types):
        v_coefficients[i][0:np.size(temp[i][0], 0)] = temp[i][0]

    # Find the method (row) with the least sum squared error
    # Ensure that there is at least a 10% decrease in error between the bettered methods, or else do not use it
    min_err_index = 0
    counter = 0
    if debug:
        print('counter:', counter, ' min_err_index: ', min_err_index)
    for error in v_errors:
        if error < v_errors[min_err_index]:
            percentage_increase = v_errors[min_err_index] - error
            if percentage_increase > standard_deviation:
                min_err_index = counter
                if debug:
                    print('% increase: ', percentage_increase)

        counter += 1
        if debug:
            print('counter:', counter, ' min_err_index: ', min_err_index)

    if debug:
        print(min_err_index)

    if not hide:
        plt.subplot()  # New subplot
        plt.title('Plotting the line of best fit graph for the given data: {}'.format(filename))
        plt.xlabel('x')
        plt.ylabel('y')

        # Plot the line accordingly between the minimum and maximum x values
        x_1_1r = x.min()
        x_1_2r = x.max()

        a = v_coefficients[min_err_index, 0]
        b = v_coefficients[min_err_index, 1]

        # 100 points is sufficiently dense
        x_range = np.linspace(x_1_1r, x_1_2r, 100)
        if debug:
            print('min err ind: ', min_err_index)

        if min_err_index == 0:
            # Sin Wave
            y_range = a + b * np.sin(x_range)

        elif min_err_index == 1:
            # Exponential curve
            y_range = a + b * np.exp(x_range)

        else:
            # Polynomial nth degree curve
            num_coefficients = min_err_index
            if debug:
                print('num coef: ', num_coefficients)
            y_range = np.zeros(np.shape(x_range))
            for i in range(num_coefficients):
                to_add = v_coefficients[min_err_index][i] * np.power(x_range, i)
                y_range = y_range + to_add

        plt.scatter(x, y, s=200, marker="x")
        plt.plot(x_range, y_range, 'r-', lw=4)

    total_error = v_errors[min_err_index]
    if debug:
        print('v_errors: ', v_errors)
        print('error for k points: ', total_error)
    return total_error


def plot_least_squares(x, y, k, hide=True, debug=False, allow_exp=False, filename=""):
    # Repeat the least squares process and plot the lines separately
    # Assumes the data is split into k - wide increments
    # 1,2,3,4,5... - polynomial
    # e - exponential
    # s - sin
    if len(x) % 20 != 0:
        print('Error, the curve is not divided into segments of 20 points.')
        exit()

    if debug:
        print('---------')

    num_lines = int(len(x) // k)  # The number of lines to divide the data into

    if not hide:
        # Hide the annoying axes deprecation messages
        warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

    total_error_global = 0
    for i in range(num_lines):
        total_error_global += minimise_error(x[i * k + 0:i * k + k], y[i * k + 0:i * k + k], hide=hide, debug=debug,
                                             allow_exp=allow_exp, filename=filename)

    print(total_error_global)
    if debug:
        print('---------')

    if not hide:
        plt.show()
    return 0


# The input arguments
arguments = sys.argv[1:]

if 3 > len(arguments) > 0:

    data_filename = arguments[0]
    data_filename_length = len(data_filename)

    if type(arguments[0]) != str:
        print('Invalid first argument type. Try running "python cw1_oa18502.py --help" for documentation.')
        exit()

    if data_filename_length < 3:
        print('Invalid first argument type. Try running "python cw1_oa18502.py --help" for documentation.')
        exit()

    # The case for which either data or -- args are supplied
    if len(arguments) == 1:
        # The acceptable cases are either: data_filename.csv or --help

        if arguments[0] == '--help':
            print(' o   Run python cw1_oa18502.py data_filename.csv to calculate the total error in the '
                  'line-approximations '
                  'across the '
                  'entire data set. \n o   Run python cw1_oa18502.py data_filename.csv --plot to calculate the '
                  'total error of '
                  'the line approximations across the entire data set and plot the lines of best fit. \n o   Run '
                  'python cw1_oa18502.py --help for the documentation again. \n o   Note that the '
                  '"data_filename.csv" file '
                  'for data must be in the same path folder as cw1_oa18502.py and must be formatted correctly. \n o   '
                  'Run python cw1_oa18502.py datafilename.csv --debug to plot with debugging in the console')
            exit()

        if data_filename[-4:data_filename_length] != '.csv':
            print('Invalid first argument type. Try running "python cw1_oa18502.py --help" for documentation.')
            exit()

        # Calculate the total error, don't plot the graph
        adv_x, adv_y = load_points_from_file(data_filename)
        plot_least_squares(adv_x, adv_y, 20, hide=True, filename=data_filename)

    elif len(arguments) == 2:
        # For the case: data_filename.csv --plot
        if data_filename[-4:data_filename_length] != '.csv':
            print('Invalid first argument type. Try running "python cw1_oa18502.py --help" for documentation.')
            exit()

        if arguments[1] == '--plot':
            # Calculate the total error, plot the graph
            adv_x, adv_y = load_points_from_file(data_filename)
            plot_least_squares(adv_x, adv_y, 20, hide=False, debug=False, filename=data_filename)
        elif arguments[1] == '--debug':
            # Calculate the total error, plot the graph WITH DEBUG
            adv_x, adv_y = load_points_from_file(data_filename)
            plot_least_squares(adv_x, adv_y, 20, hide=False, debug=True, filename=data_filename)
        else:
            print('Invalid second argument type, was expecting "--plot". Try running "python cw1_oa18502.py '
                  '--help" for documentation.')
            exit()

elif len(arguments) == 0:
    # The case for which no arguments are supplied
    print('Error, missing data argument. Try running "python cw1_oa18502.py --help" for documentation.')
    exit()
else:
    # Obscure error case, pretty much just to accommodate for argument overflow
    print('Unknown error encountered. Try running "python cw1_oa18502.py -- help" for documentation.')
    exit()
