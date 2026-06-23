Hello

In GoingToJupiterV2 you can see the code that was used to generate the results of the chapter on interplanetary transfer.

To find the optimal launch windows and demonstrate what would be the masses required I adapted the example from Tudat for MGA (Cassini). I now runs the same optimisation for everymonth and saves it to a csv file. Other files can plot the

-main_calculator.py - is the function that runs the optimisation
-runner.py - is the file where one inputs values
-verification.py, validation.py - are the file that one inputs values for verification and validation
-visualsVEE.py and visualsVVE.py - are the files to create the plots for the report with the trajectories

-deltaVplot.py - plots the different monthly delta vs for the report.