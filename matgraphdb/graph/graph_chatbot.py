from openai import OpenAI

from matgraphdb.database.neo4j.graph_database import MatGraphDB
from matgraphdb.utils import OPENAI_API_KEY


class ChatHandler:
    def __init__(self):

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        with MatGraphDB() as session:
            schema_list = session.list_schema()
            self.db_schema = "\n".join(schema_list)
        self.system_message = self.get_system_message()
        self.sample_messages = self.get_sample_messages()

        # To store each individual prompt for better context management
        self.message_history = [self.system_message]
        self.message_history.extend(self.sample_messages)

    def start_chat(self):
        print("Chat session started. Type your message:")
        while True:
            user_input = input()  # Get user input
            if user_input.lower() == "exit":  # Allow the user to exit the chat
                print("Chat session ended.")
                break

            new_messge = {'role': 'user', 'content': user_input}
            # Update the global state with the user's input
            self.update_message_history(new_messge)
            # I want a method were I feed in the current message history and it uses the openai api to generate a response
            # Generate a response using OpenAI
            response = self.generate_response(user_input)
            print("You: " + user_input)

        return None

    def update_message_history(self, new_message):
        # Append the new input to the global state
        self.message_history.append(new_message)
        return None

    def get_message_history(self):
        return self.message_history

    def get_system_message(self):
        system_prompt = f"""
You are an expert Neo4j Cypher translator who understands the question in english and convert to Cypher strictly based on the Neo4j Schema provided and following the instructions below:
1. Generate Cypher query compatible ONLY for Neo4j Version 5
2. Do not use EXISTS, SIZE keywords in the cypher. Use alias when using the WITH keyword
3. Use only Nodes and relationships mentioned in the schema
4. Always enclose the Cypher output inside 3 backticks
5. Always do a case-insensitive and fuzzy search for any properties related search. Eg: to search for a Company name use `toLower(c.name) contains 'neo4j'`
6. Candidate node is synonymous to Person
7. Always use aliases to refer the node in the query
8. Cypher is NOT SQL. So, do not mix and match the syntaxes
Schema:
{self.db_schema}
\n
"""
        system_message = {"role": "system", "content": system_prompt}
        return system_message

    def get_sample_messages(self):
        sample_messages = [
            {"role": "user", "content": "What is ?"},
            {"role": "user", "content": "Where do most candidates get educated?"},
            {"role": "user", "content": "How many people have worked as a Data Scientist in San Francisco?"}
        ]
        return sample_messages

    def generate_responce(self, user_input):
        # Construct the prompt with the latest user input
        prompt = self.system_message["content"] + \
            " ".join(self.message_history)
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "What are the materials that contains B?"},
                    {"role": "user",
                     "content": "MATCH (e:Element)-[:CONNECTS]-(m:Material) "
                     "WHERE toLower(e.elements) CONTAINS 'b' AND toLower(e.elements) CONTAINS 'n' "
                     "RETURN DISTINCT m.name AS MaterialName"},
                    {"role": "system", "content": "What are the materials that contains B?"},
                    {"role": "user",
                     "content": "MATCH (e:Element)-[:CONNECTS]-(m:Material) "
                     "WHERE toLower(e.elements) CONTAINS 'b' AND toLower(e.elements) CONTAINS 'n' "
                     "RETURN DISTINCT m.name AS MaterialName"},
                ]
            )
        except Exception as e:
            print(f"Error: {e}")
        return response


if __name__ == "__main__":
    chat_handler = ChatHandler()

    # This will print the system message
    system_message = chat_handler.get_system_message()
    system_prompt = f''
