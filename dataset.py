from sqlalchemy import create_engine
import pandas as pd

def load_and_save_csv():
    """
    This function loads the needed data from the remote postgres DB. Before using it it is required that you connect to the remote db.
    """
    # PostgreSQL connection
    db_url = "postgresql://<user>:<password>@XXX.XX.XX.XXX:PORT/aviation_db"

    engine = create_engine(db_url)

    # check tables
    query_tables = """
    SELECT *
    FROM asn_scraped_accidents;
    """

    tables = pd.read_sql(query_tables, engine)
    tables.to_csv("asn_scraped_ds.csv")

if __name__ == "__main__":
    load_and_save_csv()