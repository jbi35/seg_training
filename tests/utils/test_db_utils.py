import sqlite3
import pytest
from mock import patch
from calculon.utils.db_utils import connect_db, extract_ct_scans

def test_connect_db():
    with patch('calculon.utils.db_utils.sqlite3') as mocksql:
        conn2 = mocksql.connect()
        c2 = conn2.cursor()
        c, conn = connect_db('temp.db')
        assert c == c2
        assert conn == conn2
