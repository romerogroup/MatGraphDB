import copy

class Node:
    def __init__(self, node_name,class_name):
        self.node_name=node_name
        self.class_name=class_name
        self.create_statement="MERGE (%s:%s {name: '%s'} )" % (self.node_name, self.class_name, self.node_name)
        self.on_create_statement="ON CREATE SET %s += {" % self.node_name
        self.on_match_statement="ON MATCH SET %s += {" % self.node_name
        self.first_property=False

        self.execute_statement=copy.copy(self.create_statement)

    

    def add_property(self,name, value, put_in_string=False):
        if self.first_property:
            if value is None or put_in_string or isinstance(value,str):
                property_statement=f" {name}: '{value}' "
            else:
                property_statement=f" {name}: {value} "

            self.first_property=True
        else:
            if value is None or put_in_string or isinstance(value,str):
                property_statement=f", {name}: '{value}' "
            else:
                property_statement=f", {name}: {value} "

        self.on_create_statement+=property_statement
        self.on_match_statement+=property_statement
    
    def add_points(self,name, points):
        points_str='['
        for i,point in enumerate(points):
            points_str+='point({x: %s,y: %s,z: %s})'%(point[0],point[1],point[2])
            if i!=len(points)-1:
                points_str+=','
        points_str+=']'
        self.on_create_statement+=f", {name}: {points_str} "
        self.on_match_statement+=f", {name}: {points_str} "


    def clear(self):
        self.create_statement="MERGE (%s:%s {name: '%s'} )" % (self.node_name, self.class_name, self.node_name)
        self.on_create_statement="ON CREATE SET %s += {" % self.node_name
        self.on_match_statement="ON MATCH SET %s += {" % self.node_name
        return self.create_statement
    
    def final_execute_statement(self):
        self.on_create_statement+='}'
        self.on_match_statement+='}'

        execute_statement= self.create_statement + '\n' + self.on_create_statement + '\n' + self.on_match_statement
        return execute_statement



