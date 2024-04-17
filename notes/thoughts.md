# Thoughts



## What is the point of a database?
I can think of 2 reason, which both do not need to be true at the same time
1. Storing, maintaining, and accessing data
2. Analysis and answering questions

## How do you pick what type of database to use (Relational, Document-base, Graph)?
This depends on the point of the database. 

1. If its for storing, maintaining, and accessing data you would would pick the database which conforms more with the data you are working with. This would typically lead to smoother implementations

    - If the data is tabular I would choose relation database. An example would be a database of stocks. This can easily be stored in 

    - If the data has more complicated data types to store (dictionaries, coordinates, species), I would choose document-based database. An example would Material, which naturally has complicated datatypes to handle. 

    - If the data is relatively simple (properties are floats and strings) and is natually model interconnectively, I would choose a graph database. An example of this is a Friend network. Even then it is not common to use a graph database as the primary data store

2. If its for answering question, this depends on the type of questions we are asking

## What is type of database conforms to material data

I would say Document-based data based, as they can store more complicated data types such as coordinates, lattice, species, density of states graph, band structure graphs, etc..


## How should graph databases be incorporated into material data?

They should be used for analysis and answering question that require more complicated relationships patterns. The most ideal would be have to have a document-based database be the primary data store, then methods to import the data into graph database. 

In this data storage pattern, you would not want users to add new materials on the graph database side, you would ideally want them to add it on on your primary data store side. then use the method to import that data.

However, this does not have to be the case. Maybe the user wants to see there materials in relation to all the other materials. 

**Issues**
- Materials won't have some of the properties. (Can't use more of the complex embeddings to find similar material since this requires using other ML models to obtain them. Getting these ML models to work in single model poses a massive headache due to dependcy conflicts)
- Won't have all relationships (For example the relationship between materials and chemenv. Unless we the code calculates this for them.)

## Why let users insert materials into the graph database?

### Reasons for
- Users can see there material in relation to all the other materials

### Possible Issues
- Won't have some of the properties.
-- Can't use more of the complex embeddings to find similar material since this requires using other ML models to obtain them. Getting these ML models to work in single model poses a massive headache due to dependcy conflicts
- Won't have all relationships (For example the relationship between materials and chemenv. Unless we the code calculates this for them.)

