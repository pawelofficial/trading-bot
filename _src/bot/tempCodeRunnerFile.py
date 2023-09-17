class mydb:
    def __init__(self) -> None:
        self.db_name = "your_database_name"
        self.user = "your_username"
        self.password = "your_password"
        self.host = "localhost" # or the IP address of your PostgreSQL server
        self.port = "5432" # default PostgreSQL port

if __name__=='__main__':
    pgsql=mydb
