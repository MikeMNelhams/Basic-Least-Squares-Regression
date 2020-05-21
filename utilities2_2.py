import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib.cbook
# My own utilities library python file


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


def minimise_error(x, y, hide=True, debug=False, allow_exp=False):
    # VOID function
    num_curve_types = 7  # Number of different curves to fit
    max_coefficient = 6  # Maximum number of coefficients used in any of the algorithms

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
    for error in v_errors:
        if error < v_errors[min_err_index]:
            percentage_increase = 100 * ((v_errors[min_err_index] - error)/(v_errors[min_err_index]))
            if percentage_increase > 10:
                min_err_index = counter
                if debug:
                    print('% increase: ', percentage_increase)

        counter += 1

    if not hide:
        plt.subplot()  # New subplot
        plt.title('Plotting the line of best fits for the given data: adv_1.csv')
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
        # plt.plot(x_range, y_range, 'r-', lw=4)

    total_error = v_errors[min_err_index]
    if debug:
        print('v_errors: ', v_errors)
        print('error for k points: ', total_error)
    return total_error


def plot_least_squares(x, y, k, hide=True, debug=False, allow_exp=False):
    # Repeat the least squares process and plot the lines separately
    # Assumes the data is split into k - wide increments
    # 1,2,3,4,5... - polynomial
    # e - exponential
    # s - sin
    if len(x) % 20 != 0:
        print('Error, the curve is not divided into segments of 20 points.')
        exit()
    num_lines = int(len(x) // k)  # The number of lines to divide the data into

    if debug:
        print('---------')

    num_lines = int(len(x) // k)  # The number of lines to divide the data into

    if not hide:
        # Hide the annoying axes deprecation messages
        warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

    total_error_global = 0
    for i in range(num_lines):
        total_error_global += minimise_error(x[i * k + 0:i * k + k], y[i * k + 0:i * k + k], hide=hide, debug=debug,
                                             allow_exp=allow_exp)

    print(total_error_global)
    if debug:
        print('---------')

    if not hide:
        plt.show()
    return 0
