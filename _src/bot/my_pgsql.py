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
    def write_df(self,df,table='historical_data',if_exists='append'):
        engine = create_engine(f'postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}')
        df = df.drop_duplicates(subset=['epoch'])
        try:
            df.to_sql(table, engine, if_exists=if_exists, index=False)  # 'replace' will overwrite the table if it already exists. Use 'append' to add to an existing table.
        except Exception as er:
            print('uh oh!')
            log_stuff(msg='error in writing df',er=er,level=40)    
            self.reconnect()









if __name__=='__main__':
    p=mydb()
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