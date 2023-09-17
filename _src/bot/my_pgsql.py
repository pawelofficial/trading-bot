import psycopg2
import json 
from utils import setup_logging,log_stuff
import pandas as pd 

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
    
    def execute_dml(self,query):
        try:
            self.cur.execute(query)
            self.conn.commit()
            log_stuff(msg='dml succesfull',query=query,level=20)
        except psycopg2.Error as er:
            print('uh oh! ')
            log_stuff(msg='error in executing query',er=er,query=query, level=40)
            self.reconnect()
    def execute_select(self,query):
        try:
            self.cur.execute(query)
            records = self.cur.fetchall()

        except psycopg2.Error as er:
            print('uh oh! ')
            log_stuff(msg='error in executing query',er=er,query=query, level=40)
            self.reconnect()
            return None
        
        column_names = [desc[0] for desc in self.cur.description]
        df = pd.DataFrame(records, columns=column_names)
        
        return df 








if __name__=='__main__':
    p=mydb()
    p.execute_dml(p.queries['create_table_raw_data'])
    p.execute_dml(p.queries['truncate_table_raw_data'])
    r= p.execute_select('select current_timestamp')
    print(r)
exit(1)

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