# TUP-neediness

The data used in this project can be found at https://github.com/ecollins/TUP-data 

It's kept separately since 1) it is seldom modified, 2) contains large files,
and 3) is used by another separate project.

## Documents

Contains

- **neediness.org**, which contains the code and text of the paper. Most of the
   python files in the **analysis** folder are tangled from neediness.org.

- **HighFrequency_Consumption.org**, which contains the code to wrangle the
  mobile survey in the data repository.

## Analysis

Contains

- **TUP.py**, which contains the central functions for reading the TUP survey
  data, getting it into useable formats for asset and consumption analysis,
  running regressions using the statsmodels package, and converting dataframes
  to org tables (including regression tables with standard errors and stars).

- **estimation**, a folder including essential functions for estimating marginal
  utilities and relating them back to the Frisch demand model.
