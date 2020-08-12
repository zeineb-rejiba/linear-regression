import tensorflow as tf
import matplotlib.pyplot as plt
import argparse, sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    #define the arguments that may be given as an input to the app
    #the app will have a positional argument which is 'type', a required argument 'action' and an optional argument 'plot_type'
    parser.add_argument("type",
                        help='Regression type', choices=['univariate', 'multivariate'])
    parser.add_argument("--action", required=True, help="determines the functionality to be executed",
                        choices=['train', 'predict', 'plot', 'verify'])
    parser.add_argument("--plot_type", help="shows a plot according to the requested argument",
                        choices=['scatter', 'data-with-line', 'cost-surface-contour', 'cost-vs-iterations'])

    args = parser.parse_args()

    #retrieve arguments
    regression_type = args.type
    action = args.action
    plot_type = args.plot_type

    if action == 'plot':

        if not plot_type: #deal with the case where the user has chosen to plot but hasn't specified a plot_type.
            sys.exit('Please specify a plot type. Check python lin_reg.py -h  for help.')
        else:  # the user chose to plot something. Perform processing according to plot_type and regression_type

            if regression_type == 'univariate':
                if plot_type not in ['scatter', 'data-with-line', 'cost-surface']: # these are the plots corresponding to steps 2.1 2.2 and 2.4 of the univariate regression section
                    sys.exit('The selected plot_type is not supported for this regression type.')
                else:
                    if plot_type == 'scatter':
                        print('plotting ',plot_type)

                    elif plot_type == 'data-with-line':
                        print('plotting ', plot_type)

                    elif plot_type == 'cost-surface':
                        print('plotting ', plot_type)

            else:  # regression type is multivariate
                if plot_type != 'cost-vs-iterations': #this is the plot corresponding to step 3.2.1 of the exercise, dealing with multiple variables.
                    sys.exit('The selected plot_type is not supported for this regression type.')
                else:
                    print('plotting ', plot_type)

    else: #action is either train/predict/verify
        if plot_type: #deal with the case when the action is not plot but the user specified a plot_type.
            sys.exit('--plot_type cannot be used with an action other than "plot".')
        else:
            # perform the corresponding action
            print('performing action ', action)
