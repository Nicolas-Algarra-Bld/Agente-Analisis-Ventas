import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(3)

try:
    s.connect(("127.0.0.1", 3306))
    print("TCP connection to 127.0.0.1:3306 OK")
except Exception as e:
    print("TCP connection FAILED:", e)
finally:
    s.close()
