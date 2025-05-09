import pandas as pd
import numpy as np
import os
import re

def get_sorted_gradP_filenames(directory):
    pattern = re.compile(r"gradP_(\d+)\.csv$")
    matched_files = []

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            matched_files.append((num, filename))

    # Sort by the numeric part
    matched_files.sort(key=lambda x: x[0])

    # Return just the sorted filenames
    return [filename for _, filename in matched_files], [num for num, _ in matched_files]

if __name__ == '__main__':
    
    directory_path = os.getcwd() + '/gradP/'
    sorted_filenames, sorted_nums = get_sorted_gradP_filenames(directory_path)
    
    for i, file in enumerate(sorted_filenames):

        if i == 0:
            fullData = pd.read_csv(directory_path + file)
            numData = fullData.shape[0]
            fullData['gradP'] = sorted_nums[i]*np.ones(numData) 

        data = pd.read_csv(directory_path + file)
        numData = data.shape[0]
        data['gradP'] = sorted_nums[i]*np.ones(numData) 

        fullData = pd.concat([fullData, data], ignore_index=True)

    fullData.columns = ['uw', 'ug', 'gradP']
        
    fullData.to_csv('gradP.csv', index=False)