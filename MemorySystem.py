from py2neo import Graph
from py2neo import Node, Relationship
graph = Graph("bolt://localhost:7687", user="neo4j", password="123456")
tx = graph.begin()
a = Node("Person",name="Hani", Age="30")
tx.create(a) 
tx.commit()


######
#####MATCH (a:Person),(b:Person)
#####WHERE a.name = 'Adnan' AND b.name = 'Hani'
#####CREATE (a)-[r:husbandof]->(b)
#####RETURN type(r)