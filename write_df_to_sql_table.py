import csv
from io import StringIO

from sqlalchemy import create_engine
import psycopg2

# Import the 'config' function from the config.py file
from config import config


def psql_insert_copy(table, conn, keys, data_iter):
    """
    Execute SQL statement inserting data
    (per https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-sql-method
    and https://stackoverflow.com/questions/23103962/how-to-write-dataframe-to-postgres-table)

    Parameters
    ----------
    table : pandas.io.sql.SQLTable
    conn : sqlalchemy.engine.Engine or sqlalchemy.engine.Connection
    keys : list of str
        Column names
    data_iter : Iterable that iterates the values to be inserted
    """
    # gets a DBAPI connection that can provide a cursor
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ", ".join('"{}"'.format(k) for k in keys)
        if table.schema:
            table_name = "{}.{}".format(table.schema, table.name)
        else:
            table_name = table.name

        sql = "COPY {} ({}) FROM STDIN WITH CSV".format(table_name, columns)
        cur.copy_expert(sql=sql, file=s_buf)


def write_df_to_sql(df, table_name):

    # Create dictionary of database configuration details
    db_details = config(section="postgresql")

    user = db_details["user"]
    passw = db_details["password"]
    host = db_details["host"]
    port = db_details["port"]
    dbase = db_details["database"]

    engine = create_engine(
        "postgresql+psycopg2://"
        + user
        + ":"
        + passw
        + "@"
        + host
        + ":"
        + str(port)
        + "/"
        + dbase
    )

    df.to_sql(
        table_name, engine, chunksize=1000, if_exists="replace", method=psql_insert_copy
    )
