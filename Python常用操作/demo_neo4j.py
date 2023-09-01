from py2neo import *
import pandas as pd
import socket

graph = Graph("http://127.0.0.1:5002", auth=("neo4j", "password"))
print(graph)
