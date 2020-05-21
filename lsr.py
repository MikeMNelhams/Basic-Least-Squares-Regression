import sys
import utilities as ut1  # Utilities is the given library
import utilities2_2 as ut2  # Utilities2 is my library

# The input arguments
arguments = sys.argv[1:]

if 3 > len(arguments) > 0:

    data_filename = arguments[0]
    data_filename_length = len(data_filename)

    if type(arguments[0]) != str:
        print('Invalid first argument type. Try running "python lsr.py --help" for documentation.')
        exit()

    if data_filename_length < 3:
        print('Invalid first argument type. Try running "python lsr.py --help" for documentation.')
        exit()

    # The case for which either data or -- args are supplied
    if len(arguments) == 1:
        # The acceptable cases are either: data_filename.csv or --help

        if arguments[0] == '--help':
            print('o   Run python lsr.py data_filename.csv to calculate the total error in the line-approximations '
                  'across the '
                  'entire data set. \n o   Run python lsr.py data_filename.csv --plot to calculate the total error of '
                  'the line approximations across the entire data set and plot the lines of best fit. \n o   Run '
                  'python lsr.py --help for the documentation again. \n o   Note that the "data_filename.csv" file '
                  'for data must be in the same path folder as lsr.py and must be formatted correctly.')
            exit()

        if data_filename[-4:data_filename_length] != '.csv':
            print('Invalid first argument type. Try running "python lsr.py --help" for documentation.')
            exit()

        # Calculate the total error, don't plot the graph
        adv_x, adv_y = ut1.load_points_from_file(data_filename)
        ut2.plot_least_squares(adv_x, adv_y, 20, hide=False)

    elif len(arguments) == 2:
        # For the case: data_filename.csv --plot
        if data_filename[-4:data_filename_length] != '.csv':
            print('Invalid first argument type. Try running "python lsr.py --help" for documentation.')
            exit()

        if arguments[1] == '--plot':
            # Calculate the total error, plot the graph
            adv_x, adv_y = ut1.load_points_from_file(data_filename)
            ut2.plot_least_squares(adv_x, adv_y, 20, hide=False, debug=False)
        else:
            print('Invalid second argument type, was expecting "--plot". Try running "python lsr.py --help" for '
                  'documentation.')
            exit()

elif len(arguments) == 0:
    # The case for which no arguments are supplied
    print('Error, missing data argument. Try running "python lsr.py --help" for documentation.')
    exit()
else:
    # Obscure error case, pretty much just to accommodate for argument overflow
    print('Unknown error encountered. Try running "python lsr.py -- help" for documentation.')
    exit()


