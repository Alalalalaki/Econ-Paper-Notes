"""
This file replicates the code in march-cleaners.
The input is raw-data. The output is cleaned-data.

# means original comment
#@ means additional comment
"""

import pandas as pd

input_path = "../../ref/origin/March-CPS/raw-data/"

"""
clean7678km.do
"""

## Initial assembly of 1976-78 data, prior to hours imputation calcs

# The variable names and codes differ across years.  This file makes the years 1976-78
# comparable to all other years.  The years 1976-78 are then combined into one file found at

inputs = [input_path+"mar"+str(y)+".dta" for y in range(76, 79)]

for i in inputs:

    # exclude those individuals in the armed forces (AF) because in some years certain variables
    # are not asked of those in AF(i.e. "hours" in 1962).  Keep individuals who worked at least one
    # week last year and who were at least age 16 LAST YEAR(age >= 17).
    pass


"""
clean6275km.do
"""








"""
clean7909km.do
"""
