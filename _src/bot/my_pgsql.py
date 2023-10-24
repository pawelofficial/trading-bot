import psycopg2
import json 
from utils import setup_logging,log_stuff
import pandas as pd 
from sqlalchemy import create_engine

class mydb:
    def __init__(self) -> None:
        setup_logging(name='pgsql',mode='w')
        log_stuff(msg='starting pgsql ')
        self.db_name = "dev"
        self.user = "postgres"
        self.password = "admin"
        self.host = "127.0.0.1" # or the IP address of your PostgreSQL server
        self.port = "5432" # default PostgreSQL port

        self.conn=None
        self.connect()
        self.cur = self.conn.cursor()
        
        self.queries=json.loads(open('pgsql_queries.json').read()) 




    # connects to pgsql 
    def connect(self):
        self.conn=psycopg2.connect(
            dbname=self.db_name,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )
        
    # closes and opens connection and cursor 
    def reconnect(self):
        self.conn.close()
        self.cur.close()
        self.connect()
        self.cur = self.conn.cursor()        

    # pings pgsql 
    def ping(self):
        self.cur.execute('select current_timestamp')
        records = self.cur.fetchall()
        return records 
    
    # executes dml
    def execute_dml(self,query):
        try:
            self.cur.execute(query)
            self.conn.commit()
            log_stuff(msg='dml succesfull',query=query,level=20)
        except psycopg2.Error as er:
            print('uh oh! ')
            print(er)
            log_stuff(msg='error in executing query',er=er,query=query, level=40)
            self.reconnect()
    # executes sql 
    def execute_select(self,query,nrows=-1):

        try:
            self.cur.execute(query)
            if nrows==-1:
                records = self.cur.fetchall()
            else:
                records = self.cur.fetchmany(nrows)
        except psycopg2.Error as er:
            print('uh oh! ')
            print(er)
            log_stuff(msg='error in executing query',er=er,query=query, level=40)
            self.reconnect()
            return None
        column_names = [desc[0] for desc in self.cur.description]
        df = pd.DataFrame(records, columns=column_names)
        
        return df 
    
    # writes df to a db ! 
    def write_df(self,df,table='historical_data',if_exists='append',cols_list=[],deduplicate_on=None):
        engine = create_engine(f'postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}')

        if deduplicate_on is not None:
            df=df.drop_duplicates(subset=deduplicate_on)

        if cols_list==[]:
            df = df
        else:
            df=df[cols_list]

        try:
            df.to_sql(table, engine, if_exists=if_exists, index=False)  # 'replace' will overwrite the table if it already exists. Use 'append' to add to an existing table.
        except Exception as er:
            print('uh oh!')
            print(er)
            log_stuff(msg='error in writing df',er=er,level=40)    
            self.reconnect()

    # writes df to a db converting bunch of columns into array type in a db ! super slow but it works ! 
    def write_df_array(self
                       ,df
                       ,tbl='quantiles'        # table where we insert 
                       ,tgt_ar_col='ar'        # columns name of array type on postgresql  
                       ,df_ar_cols=None        # explicit specify which df columns should go to array 
                       ,df_base_cols=[]        # specify which columns are not arrays , CANT BE EMPTY ! 
                       ,truncate_load=True     # truncate load 
                       ):     # specify which columns should not go to array type 
        if truncate_load:
            self.execute_dml(f'truncate table {tbl}')        

        if df_ar_cols is None: # if provided df array columns is none - use all columns except base columns
            df_cols=df.columns.tolist()
            df_ar_cols=[x for x in df_cols if x not in df_base_cols]
        else:                  # else set df array columns to provided df array columns
            df_ar_cols=df_ar_cols
        
        cols=df_base_cols.copy()
        cols.append(tgt_ar_col)
        cols=','.join(cols)
        quote="'"
        LAMBDA_INSERT=lambda tbl,cols,vals : f'INSERT INTO {tbl}({cols}) VALUES ({vals});'
        for no,row in df.iterrows():
            row_d=row.to_dict()
            ar=[str(row_d[x]) for x in df_ar_cols]
#            base_col_vals=','.join([f"{row_d[x]}" for x in df_base_cols])
            base_col_vals=','.join([f"'{row_d[x]}'" for x in df_base_cols])
            ar_vals='ARRAY['+','.join([f"{x}" for x in ar])+']'
            s=base_col_vals + ', ' + ar_vals 
            
            if base_col_vals=='':
                vals=ar_vals
            else:
                vals=', '.join([base_col_vals, ar_vals])
            INSERT=LAMBDA_INSERT(tbl,cols,vals)
            #print(INSERT)
            #input('wait')
            self.execute_dml(query=INSERT)
            
    # reads a table from postgresql with an array type into a dataframe splitting the array into columns 
    def read_df_array(self 
                      ,tbl='quantiles'            # table we select from  
                      ,tgt_ar_col='ar'            # column name of array type, rest goes to df no cap 
                      ,last_n_rows=5              # fetch last n rows order by id
                      ,order_by = 'id desc'       # order by clause 
                      ):
        
        
        if last_n_rows is not None:
            query = f'SELECT * FROM {tbl} order by {order_by} limit {last_n_rows} '
        else:
            query = f'SELECT * FROM {tbl} '
        df = self.execute_select(query)         
        ar_df=pd.DataFrame(df[tgt_ar_col].to_list(), columns=[f"ar_{i}" for i in range(len(df[tgt_ar_col][0]))])
        # Drop the original 'ar' column from df
        df.drop(columns=[tgt_ar_col], inplace=True)
        # Concatenate the original df and the new ar_df
        df = pd.concat([df, ar_df], axis=1)
        # remove index column from df 
        return df.reset_index(drop=True)
    
    def df_to_ddl(self
        ,df
        ,table_name
        ,col_list=[]             # specify your columns or use all cols from df 
        ,extra_cols=[]           # extra columns you wish to have 
        ,add_id=True             # adds surrogate key 
        ):
        
        sql_type_mapping = {
                "int64": "INTEGER",
                "float64": "FLOAT",
                "object": "TEXT",  # assuming object type implies string type
                "bool": "BOOLEAN",
                # Add more Pandas-to-SQL dtype mappings as needed
            }
        columns_ddl = []
        if add_id:
            columns_ddl.append("id SERIAL PRIMARY KEY")
        
        if col_list==[]:
            cols = df.columns
        else:
            cols= col_list
        
        for col_name, dtype in zip(cols, df.dtypes):
            sql_dtype = sql_type_mapping.get(str(dtype), "TEXT")  # default to TEXT if dtype not mapped
#            columns_ddl.append(f"{col_name} {sql_dtype}")
            columns_ddl.append(f'"{col_name}" {sql_dtype}')

        for tup in extra_cols:
            extra_colname=tup[0]
            data_type=tup[1]
            columns_ddl.append(f'"{extra_colname}" {data_type}')
        

        columns_ddl_str = ", ".join(columns_ddl)

        ddl = f"CREATE TABLE {table_name} ({columns_ddl_str});"
        return ddl

    # updates value in a table 
    def update_val(self
                   ,tbl='test'             # which table 
                   ,where_col='id'         # where column
                   ,where_val='1'          # where value 
                   ,val='test'             # value to update 
                   ):
        query=f"update {tbl} set {where_col}='{val}' where {where_col}='{where_val}'"
        pass


def test__read_df_array():
    p=mydb()
    df=p.read_df_array()
    print(df.columns)
    print(len(df))
    print(df)
    exit(1)


if __name__=='__main__':
    test__read_df_array()
    p=mydb()
    p.execute_dml(p.queries['create_table_quantiles'])
    exit(1)
    df=pd.read_csv('./data/quantiles_df.csv',sep='|')
    ar_columns=[x for x in df.columns.tolist() if 'q' in x]
    p.write_df_array(df=df,tgt_ar_col='ar',tbl='quantiles',df_ar_cols=ar_columns)
    exit(1)

    
    df=p.read_df_array(df_ar_cols_names=ar_columns)
    print(df.head())
    
    
    exit(1)
    if 0:
        p.execute_dml(p.queries['create_table_live_data'])
        #input('wait')
        p.execute_dml(p.queries['create_table_historical_data'])
        #input('wait')
        p.execute_dml(p.queries['truncate_table_live_data'])
        #input('wait')
        p.execute_dml(p.queries['truncate_table_historical_data'])
        #input('wait')
        p.execute_dml(p.queries['create_vw_data'])
        p.execute_dml(p.queries['create_vw_agg5'])

    exit(1)
    
    df=pd.read_csv('./data/data.csv',sep='|')#.iloc[:100]
    p.write_df(df=df,table='historical_data',if_exists='append')

    
#  1. load historical data to pgsql 
#  2. load live data to pgsql 
#  3. have dedupliacted view 
#  4. have agg view 
#  5. once agg view count changes / other condition - evaluate model 




###
###class mydb:
###    def __init__(self):
###
###    
###    conn = psycopg2.connect(
###        dbname=db_name,
###        user=user,
###        password=password,
###        host=host,
###        port=port
###    )
###
###    # Create a cursor object to interact with the database
###    cur = conn.cursor()
###
###    # Example query
###    cur.execute("SELECT * FROM your_table_name;")
###    records = cur.fetchall()
###    print(records)
###
###    # Close the cursor and connection
###    cur.close()
###    conn.close()
###