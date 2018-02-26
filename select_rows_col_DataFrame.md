# How to select rows or columns from a DataFrame of Pandas?
## 1. Row selection
### a. Slicing one line or multiple lines.
suppose df is a DataFrame object. 
sub_df=df[1:3] # The same with python list slicing, it starts at 0 and select
row1 and row2 (not include row 3).
### b. Using index to select one row or multiple rows.
#### a_series=df.loc[1] # selecting the row with index 1. If the DataFrame has no index,
then use 0,1,2...N by default.
#### sub_df=df.loc[1:3] # It's not slicing,  so it will take row 1, row 2 and row 3.
it uses the indexes to select 3 rows.
#### sub_df=df.loc[[1,3]] # selecting the row 1 and row 3. Note that [1,3] is a list.
### c. Selectiong rows according to their location (row number).
For DataFrame that doesn't setup index, using iloc[] and loc[] are the same, since 
the number starts at 0 and increase by 1 in both cases.
Note: Using df.loc[2] incurs error if the there is no row with index 2.

Sum up:
#### 1. loc[index], index can be string and integer. Purely label-location based
indexer for selection by label.
#### 2. iloc[rowNumber], row number must be an integer. Purely integer-location
based indexing for selection by position

## 2. Column selection
Column selection is simple, just pass in the column_name, for selectin multiple 
columns, just list the column names or column indexes.
a_series= df.[column_name]
### a. Using column names: 
sub_df=df.[['col_name_1','col_name_2']],['col_name_1','col_name_2'] is a list.
### b. Using column indexes:
sub_df=df.[[0,1,2]]: Using column index to select multiple columsn.
This one is very confusing. In fact, the every column has a default column index.
if you want to select first 5 columns: sum_df=df.[[0:5]] incurs error.
You should create a list in the parameter: sub_df=df.[list(range(5))]

