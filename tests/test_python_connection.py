import mysql.connector
import os

try:
    conn = mysql.connector.connect(
        host="127.0.0.1",   # NOT localhost
        port=3306,
        user="user", # Use your user name
        password="password", # Use your user password
        connection_timeout=5
    )
    print("MySQL connection OK")
    conn.close()
except mysql.connector.Error as err:
    print("MySQL error:", err)
