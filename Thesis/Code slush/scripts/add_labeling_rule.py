import edec.afterprocessing as ap
import sqlite3
import numpy as np
import edec.preprocessing as pp

# Sql which data is taken from
link = "df.db"

# Name of SQL table you want to create
table_name = "cell_fail"

sql = "SELECT [rid],[Volt_dev],[Soc_Dev] FROM main_table"
df = ap.load_sql(sql, db=link)

threshold_soc = 9
threshold_volt = 0.01

# Label fails
df["Fail"] = np.where(pp.label_data(
    df, threshold_soc, threshold_volt)["Label"], 1, 0)
df["Fail_Soc"] = np.where(pp.label_data(df, threshold_soc, 0)["Label"], 1, 0)
df["Fail_Volt"] = np.where(pp.label_data(df, 0, threshold_volt)["Label"], 1, 0)

# Connect to SQL
conn = sqlite3.connect("../scripts/" + link)
df.to_sql(table_name, con=conn, if_exists='replace')
conn.close()
