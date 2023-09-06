# Point-to-Function Allocation (P2FA)

## Overview

This project aims to allocate data points to their closest ideal functions among a set of given ideal functions. The main goal is to determine the largest deviations between training functions and ideal functions, and then assign test data points to the most suitable ideal functions.

## Features

- **Preselect Functions**: Determines the closest matches between the training functions and ideal functions based on the least deviation.
  
- **Map Points to Functions**: Assigns test data points to the closest ideal functions, based on their calculated deviation.

- Identifies the largest deviations between training and ideal functions

- Maps test data points to the ideal functions

- Includes unit tests

- Uses a MySQL database for data storage

- Bokeh for interactive visualizations

## File Overview

- `database.py`: Responsible for managing the data storage, including reading and writing data sets like test data, training data, and ideal function data.

- `p2f_alloc.py`: Contains the main implementation logic for the Point-to-Function Allocator, including the algorithms to preselect functions and map points to functions.

- `p2fa.py`: The entry point to the application, where the data is loaded and the main workflow is orchestrated.

- `test_p2fa.py`: Includes unit tests for the project to validate functionality.

- `README.md`: This file, explaining the project, how to install it, and how to use it.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/patdring/p2fa.git
    ```

2. Navigate to the project directory:
    ```sh
    cd p2fa
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Testing

This project includes unit tests to ensure that the implemented functions work as expected. To run the tests, execute:

```sh
python -m unittest test_p2fa.py
```

The tests are located in `test_p2fa.py` and they cover the following functionalities:

- `test_greatestDeviations`: Tests whether the largest deviations between training and ideal functions are correctly determined and assigned.
  
- `test_mapPoints2Functions`: Checks if test data points are correctly assigned to an ideal function based on their deviation.

## Usage

```python
import p2f_alloc as p2fa
import pandas as pd

# Create a DataFrame for your test and training data
df_testData = pd.DataFrame({ ... })
df_trainingData = pd.DataFrame({ ... })
df_idealData = pd.DataFrame({ ... })

# Initialize the Point-to-Function Allocator
allocator = p2fa.CPoint2FunctionAllocator()

# Get the best matching functions and greatest deviations
matches, greatestDeviations = allocator.preselectFunctions(df_trainingData, df_idealData)

# Map the test points to the ideal functions
resultTable = allocator.mapPoints2Functions(df_testData, df_idealData, matches, greatestDeviations)
```

