

#-------------------------------------#
#
# 6 NOV 2019
# Aleksander Frese (s163859)
# Migrating Terrorist data to Grakn
# Input format: CSV
#
# This migration script is for the
# FULL terrorist network (both persons
# and relations between them).
#
# This file assmues that it is placed
# in a different subfolder than the 
# csv files.
#
#	- root
#		- CsvFiles
#			+ nodes_geo_graph.csv
#			+ ...
#		- grakn
#			+ migrate-entities-only.py
#			+ ...
#
#-------------------------------------#


from grakn.client import GraknClient
import csv
import os
from pathlib import Path

host = 'localhost'
port = '48555'
keyspace = 'terrorist_network'

cwd = os.getcwd()
#parent = Path(cwd).parent
dir_csvfiles = str(cwd) + '/CsvFiles/'

# creates a keyspace for the graph if it does not already exist
# loops over the elements in the input dictionary
def build_graph(inputs):
	with GraknClient(uri=host+":"+port) as client: # connect to local grakn server
		with client.session(keyspace=keyspace) as session: # connect to terrorist keyspace, if it does not exist, then create it
			#for input in inputs:
				#print("Loading from [" + input["data_path"] + "] into Grakn ...")
				#print("input =>", input) # debugging
				#load_data_into_grakn(input, session) # start loading the actual data into the keyspace and network
			
			## testing ##
			for c, input in enumerate(inputs,1):
				#if c == 1: # skip person csv. only test creating the edges
					#continue
				print("Loading from [" + input["data_path"] + "] into Grakn ...")
				#print("input =>", input) # debugging
				load_data_into_grakn(input, session) # start loading the actual data into the keyspace and network


# loads one csv file into the grakn keyspace
def load_data_into_grakn(input, session):
	items = parse_data_to_dictionaries(input) # input: path to csv file and function to use as template for generating gql insert statements, output: list of dictionaries containing each of the rows from the csv
	#print() # debugging
	#print("converted items =>", items) # debugging

	for item in items:
		with session.transaction().write() as transaction:
			graql_insert_query = input["template"](item)
			print("Executing Graql Query: " + graql_insert_query)
			transaction.query(graql_insert_query)
			transaction.commit()
			#break # debugging, only checking the graql insert statement for one row

	print("\nInserted " + str(len(items)) + " items from [ " + input["data_path"] + "] into Grakn.\n")


# helper function for csv format
# input format:
# output format: 
def parse_data_to_dictionaries(input):
	items = []
	with open(input["data_path"] + ".csv") as data:
		#print(data) # debugging
		for row in csv.DictReader(data, skipinitialspace=True):
			#print(row) # debugging
			item = { key: value for key, value in row.items() }
			items.append(item)
	return items


# insert person
def person_template(person):
	graql_insert_query = 'insert $person isa person, has tID "' + person["Terrorist Id"] + '"'
	graql_insert_query += ', has name "' + person["Terrorist Name"] + '"'
	graql_insert_query += ', has nationality "' + person["Nationality"] + '"'
	graql_insert_query += ', has affiliation "' + person["Affilation"] + '"'
	#graql_insert_query += ', has latitude ' + str(person["Latitude"]) + ''
	#graql_insert_query += ', has longitude ' + str(person["Longitude"]) + ''
	graql_insert_query += ";"
	return graql_insert_query


def connection_template(connection):
	# match source person
	graql_insert_query = 'match $source isa person, has tID "' + connection["Source"] + '";'
	# match target person
	graql_insert_query += ' $target isa person, has tID "' + connection["Target"] + '";'

	# insert connection
	if connection["family"] == "1":
		graql_insert_query += " insert $relation(relative: $source, relative: $target) isa Family;"
	elif connection["colleague"] == "1":
		graql_insert_query += " insert $relation(colleague: $source, colleague: $target) isa Organisation;"
	elif connection["congregate"] == "1":
		graql_insert_query += " insert $relation(participant: $source, participant: $target) isa Congregation;"
	elif connection["contact"] == "1":
		graql_insert_query += " insert $relation(contactor: $source, contactee: $target) isa Contact;"

	graql_insert_query += ' $relation has rel_url "' + connection["rel_url"] +'";'
	return graql_insert_query


inputs = [
	{
		"data_path": dir_csvfiles + 'nodes_geo_graph',
		"template": person_template
	},
	{
		"data_path": dir_csvfiles + 'edges_geograph_dual_graph',
		"template": connection_template
	}

]


build_graph(inputs)
















