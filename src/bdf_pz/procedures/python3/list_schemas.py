import pandas as pd
from prettytable import PrettyTable

# construct table for printing
table = [["Name", "Fields"]]
for name, fields in existing_schemas.items():
    field_names = ", ".join([field["name"] for field in fields])
    table.append([name, field_names])

# print table of registered datasets
t = PrettyTable(table[0])
t.add_rows(table[1:])
t