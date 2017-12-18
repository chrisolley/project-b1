# -*- coding: utf-8 -*-

def read_data(N):
    
    '''
    read_data: reads data from the lifetime.txt text file.
    Args: 
        N: number of data points to take (in chronological order).
    Returns:
        lifetime: array of measurements of lifetime.
        uncertainty: array of associated uncertainty for lifetime measurements.
    '''

    f = open('lifetime.txt')  # measured lifetimes and uncertainty data
    lifetime = [] # initialise arrays to store extracted data
    uncertainty = []
    
    for i, line in enumerate(f):  # reading data line by line
        data = line.split() # split each line into required values
        if (i < N): # only read N values
            lifetime.append(float(data[0]))
            uncertainty.append(float(data[1]))
        else: 
            break
        
    return lifetime, uncertainty