#-------------------------------------#
#
# 16 NOV 2019
# Aleksander Frese (s163859)
# Delete terrorist data from Grakn
#
# Note: this only removes the data in the
# keyspace, not the keyspace itself. We
# want to retain the keyspace in case
# there are saved/custom queries.
#
#-------------------------------------#

from grakn.client import GraknClient

host = 'localhost'
port = '48555'
keyspace = 'terrorist_network'

delete_entities = 1
delete_relations = 1

with GraknClient(uri=host+":"+port) as client:
	with client.session(keyspace=keyspace) as session:
			with session.transaction().write() as trans:
				if delete_entities:
					query = 'match $p isa person; delete;'
					trans.query(query)
					#trans.commit()
					print("Deleted entities")
				if delete_relations:
					query = 'match $r isa relation; delete;'
					trans.query(query)
					#trans.commit()
					print("Deleted relations")
				trans.commit()