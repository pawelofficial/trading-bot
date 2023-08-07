#https://community.snowflake.com/s/article/Importing-3rd-party-Python-packages-containing-non-Python-files-using-Snowflake-stages-when-creating-Python-UDFs
from snowflake.snowpark import Session
import torch 

connection_parameters = {
      "account": "jobvmbu-cq00542"
      ,"database": "DEV"
      ,"password": '!QA2ws3ed'
      ,"role": "ACCOUNTADMIN"
      ,"schema": "DEV"
      ,"threads": 1
      ,"type": "snowflake"
      ,"user": "MUSK43362"
      ,"warehouse": "COMPUTE_WH"
}

execute_sql = lambda s: print(session.sql(s).collect())               # executes sql on sf and prints the result


session = Session.builder.configs(connection_parameters).create()
session.add_packages('snowflake-snowpark-python','pytorch')


execute_sql("create or replace stage mystage")


session.file.put('./file.txt','@mystage',overwrite=True,auto_compress=False )
session.file.put('./model.pth','@mystage',overwrite=True,auto_compress=False )



#df = session.create_dataframe([[1, 2], [3, 4]], schema=["a", "b"])
#df = df.filter(df.a > 1)
#df.show()
#pandas_df = df.to_pandas()  # this requires pandas installed in the Python environment
#result = df.collect()