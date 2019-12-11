
#-------------------------------------#
#
# 6 NOV 2019
# Aleksander Frese (s163859)
# Exporting terrorist data from Grakn
# Output format: csv file.
#
# This exports persons, their attributes
# as well as relations.
#
#-------------------------------------#

from grakn.client import GraknClient
import csv
import re

host = 'localhost'
port = '48555'
keyspace = 'terrorist_network'

# what to export. 0 = dont export, 1 = export
entities = 1
relations = 1

with GraknClient(uri=host+":"+port) as client:
	with client.session(keyspace=keyspace) as session:
		# export entities (instances of the person concept and its attributes)
		if entities:
			with session.transaction().read() as read_transaction:
				#query = 'match $p isa person, has name $n; get; sort $n asc; limit 100;' # get id, name
				#query = 'match $p isa person, has name $n, has nationality $x; get; sort $n asc; limit 100;' # get id, name and nationality
				query = 'match $p isa person, has tID $id, has name $nam, has nationality $nat, has affiliation $aff; get; sort $nam asc; limit 100;' # get id, name, nationality and affiliation
				
				answers = read_transaction.query(query)
				result = []
				for answer in answers:
					answer = answer.map()
					#print(answer)
					result.append([
						#dir(answer.get('nam'))
						
						answer.get('p').id,
						answer.get('id').value(),
						answer.get('nam').value(),
						answer.get('nat').value(),
						answer.get('aff').value()
						

					])
					#break # debugging
				#  print(result) # debugging

				# export to csv file
				with open('./Exports/grakn-exported-entities.csv', 'w', newline='') as f:
					writer = csv.writer(f)
					writer.writerow(['Grakn ID', 'Terrorist ID', 'Name', 'Nationality', 'Affiliation']) # insert header row
					writer.writerows(result)

				print('Successfully exported', len(result), 'entities')

		# export relations
		if relations:
			with session.transaction().read() as read_transaction:

				# ---- query 1 - find relevant relations by looking at the grakn id (must have format V....)
				#query = 'match $s sub relation; get;'
				query = 'match $s isa relation; get;'
				answers = read_transaction.query(query)
				relations = []
				pattern = "[V][0-9]+"
				for answer in answers:
					answer = answer.map()
					if re.search(pattern, answer.get('s').id): # check if the returned grakn id is of form V.... - if it is, then we want it
						relations.append([
							answer.get('s').id,
						])
						#break # debugging
				#print(relations) # debugging

				result = []
				for id in relations:
					query = 'match $x id %s; get;' % id[0]
					#query = 'match $x id V40984; get;' # testing
					#print(query) # debugging
					answers = read_transaction.query(query)
					for answer in answers:
						answer = answer.map()
						answer = answer.get('x')
						#answer = dir(answer.type())
						strtype = answer.type().label() # get relation type, i.e. Organisation, Contact, etc.
						
						roleplayers = answer.role_players()
						answer1 = next(roleplayers) # get first person in the relation
						answer2 = next(roleplayers) # get second person in the relation					
						answer1_attr = answer1.attributes()
						answer2_attr = answer2.attributes()

						tID1 = next(answer1_attr).value()
						while tID1.isdigit() == False:
							tID1 = next(answer1_attr).value()

						tID2 = next(answer2_attr).value()
						while tID2.isdigit() == False:
							tID2 = next(answer2_attr).value()
						
						#rint("relation type:", strtype)
						#print("terrorist 1 id:", tID1)
						#print("terrorist 2 id:", tID2)
						#print(answer1.id) # this is the grakn id - we need the tID
						#print(answer2.id) # this is the grakn id - we need the tID

						result.append([tID1, tID2, strtype])

					#break # debugging


				# export to csv file
				with open('./Exports/grakn-exported-relations.csv', 'w', newline='') as f:
					writer = csv.writer(f)
					writer.writerow(['Source', 'Target', 'Relation type'])
					writer.writerows(result)

				print('Successfully exported', len(result), 'relations')