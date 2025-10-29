import pandas as pd
from prettytable import PrettyTable
from prettytable import TableStyle

import os 

# construct table for printing
table = [["Name", "Path", "N. Files"]]
for name, path in registered_datasets.items():
    try:
        n_files = len(os.listdir(path))
    except:
        n_files = "1"
    table.append([path, name, n_files])

# print table of registered datasets
t = PrettyTable(table[0])
t.add_rows(table[1:])
t.set_style(TableStyle.MARKDOWN)
t