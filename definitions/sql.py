from dataclasses import dataclass
from prompts import FewShotPrompt

TABLES = {
    "restaurant":"""
CREATE TABLE restaurant(
  name text,
  food text,
  pricerange text CHECK (pricerange IN (dontcare, cheap, moderate, expensive)),
  area text CHECK (area IN (centre, east, north, south, west)),
  booktime text,
  bookday text,
  bookpeople int
)
/*
5 example rows:
SELECT * FROM restaurant LIMIT 5;
name  food  pricerange  area  booktime bookday  bookpeople
pizza hut city centre italian dontcare centre  13:30 wednesday 7
the missing sock  international moderate  east  dontcare dontcare  2
golden wok chinese moderate north 17:11 friday 4
cambridge chop house  dontcare  expensive  center 08:43 monday  5
darrys cookhouse and wine shop  modern european expensive center  11:20 saturday  8
*/
""",


    "hotel":"""
CREATE TABLE hotel(
  name text,
  pricerange text CHECK (pricerange IN (dontcare, cheap, moderate, expensive)),
  type text CHECK (type IN (hotel, guest house)),
  parking text CHECK (parking IN (dontcare, yes, no)),
  bookstay int,
  bookday text,
  bookpeople int,
  area text CHECK (area IN (dontcare, centre, east, north, south, west)),
  stars int CHECK (stars IN (dontcare, 0, 1, 2, 3, 4, 5)),
  internet text CHECK (internet IN (dontcare, yes, no))
  )
/*
4 example rows:
SELECT * FROM hotel LIMIT 4;
name  pricerange  type  parking bookstay bookday  bookpeople area  stars internet
express by holiday inn cambridge  dontcare  guest house yes 3 monday  2 east  dontcare  no
a and b guest house moderate  guest house  dontcare  3 friday  5 east  4 yes
ashley hotel  expensive hotel yes 2 thursday  5 north 5 yes
el shaddia guest house  cheap guest house  yes 5 friday  2 centre  dontcare  no

*/
""",


    "train":"""
CREATE TABLE train(
  destination text,
  departure text,
  day text,
  bookpeople int,
  leaveat text,
  arriveby text
)
/*
3 example rows:
SELECT * FROM train LIMIT 3;
destination departure day bookpeople leaveat arriveby
london kings cross  cambridge monday  6 dontcare 05:51
cambridge stansted airport  dontcare  1 20:24 20:52
peterborough  cambridge saturday  2  12:06  12:56
*/
""",


    "taxi":"""
CREATE TABLE taxi(
  destination text,
  departure text,
  leaveat text,
  arriveby text
)
/*
3 example rows:
SELECT * FROM taxi LIMIT 3;
destination departure leaveat arriveby
copper kettle royal spice 14:45 15:30
magdalene college  university arms hotel dontcare  15:45
lovell lodge  da vinci pizzeria 11:45 dontcare
*/
""",


    "attraction":"""
CREATE TABLE attraction(
  name text,
  area text CHECK (area IN (dontcare, centre, east, north, south, west)),
  type text CHECK (type IN (architecture, boat, church, cinema, college, concert hall, entertainment, hotspot, multiple sports, museum, nightclub, park, special, swimming pool, theatre))
)
/*
4 example rows:
SELECT * FROM attraction LIMIT 4;
name area type
abbey pool and astroturf pitch  centre  swimming pool
adc theatre centre  theatre
all saints church dontcare  architecture
castle galleries  centre  museum
*/
"""
}

####################
# FEW-SHOT PROMPTS
####################

FEW_SHOT_STATE_PROMPT = """
Write a valid SQL query to extract the information from the Table given the customer's request. Make sure to end with a semicolon.
Focus only on the values mentioned in the last utterance.
------
{}{}
---------
Now complete the following example:
Context: 
{}
Customer: {}
"""

def state_system_prompt(domain):
    return f"""
You are a task-oriented conversational AI assistant that helps users to book {domain}.
Using valid SQLite, answer the following multi-turn conversational questions for the table provided below."""# sql table will be concatenated after, see model.py


FEW_SHOT_RESPONSE = {
    "restaurant": 
"""
Definition: You are an assistant that helps people to book a restaurant.
You can search for a restaurant by area, food, or pricerange.
There is also a number of restaurants in the database currently corresponding to the user's request.
If multiple restaurants are available, the Assistant should ask for further preferences. 
If you find a possible restaurant, the Assistant should provide [restaurant_name], [restaurant_address], [restaurant_phone] or [restaurant_postcode] if asked. Use these exact placeholders.
If no restaurants are available, the Assistant should ask for differrent preferences.
Before booking a table, the Assistant should ask for the time, the day and number of people. 
The Assistant should provide [reference] when the booking has been made. Use these exact placeholders.
**Always act as if booking is available.**
Write the Assistant response as a single line, based on the state and the database. Act friendly and engaging.
------
{}{}
---------
Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
Assistant:""",

    "hotel": 
"""
Definition: You are an assistant that helps people to book a hotel.
The customer can ask for a hotel by name, area, parking, internet availability, or pricerange.
There is also a number of hotels in the database currently corresponding to the user's request.
If you find a hotel, the Assistant should provide [hotel_name], [hotel_address], [hotel_phone] or [hotel_postcode] if asked. Use these exact placeholders.
Before booking a hotel, the Assistant should ask for the length of the stay, the day of the booking and the number of people. 
The Assistant should provide [reference] when the booking has been made. Use these exact placeholders.
**Always act as if booking is available.**
Write the Assistant response as a single line, based on the state and the database. Act friendly and engaging.
------
{}{}
---------
Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
Assistant:""",

    "train":
"""
Definition: You are an assistant that helps people to find a train connection.
The customer needs to specify the departure and destination stations, and the time of departure or arrival.
There is also a number of trains in the database currently corresponding to the user's request.
If you find a train, the Assistant should provide [arriveby], [leaveat], [departure] or [duration] if asked. Use these exact placeholders.
Before booking a train, the Assistant should ask for the day of the booking and the number of people. 
The Assistant should provide [reference] when the booking has been made. Use these exact placeholders.
**Always act as if booking is available.**
Write the Assistant response as a single line, based on the state and the database. Act friendly and engaging.
------
{}{}
---------
Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
Assistant:""",
    
        "taxi":
"""
Definition: You are an assistant that helps people to book a taxi.
The customer needs to specify the time of departure or of arrival as well as the departure and destination. 
The Assistant should provide the type of the car as [type] and [phone] as the phone number in the answer. Use these exact placeholders.
**Always act as if booking is available.**
Write the Assistant response as a single line, based on the state. Act friendly and engaging.
------
{}{}
---------
Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
Assistant:""",

    "attraction":
"""
Definition: You are an assistant that helps people to find an attraction.
The customer can ask for an attraction by name, area, or type.
There is also a number of attractions provided in the database.
If you find an attraction, the Assistant should provide [attraction_name], [attraction_address], [attraction_phone] or [attraction_postcode] if asked. Use these exact placeholders. 
The Assistant should also provide the [entrancefee] to the attraction if asked. Use this exact placeholder.
Write the Assistant's response as a single line, based on the state and the database. Act friendly and engaging.
------
{}{}
---------
Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
Assistant:""",

    "hospital":
"""
Definition: You are an assistant that helps people to find a hospital. 
Write the Assistant response as a single line, based on the state. Act friendly and engaging.
------
{}{}
---------
Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
Assistant:""",
}

@dataclass
class FewShotRestaurantDefinition:
    state_prompt = FewShotPrompt(
        template=FEW_SHOT_STATE_PROMPT,
        args_order=["history", "utterance"],
        system_prompt=state_system_prompt("restaurants"),
        sql_table=TABLES["restaurant"]
    )    

    response_prompt = FewShotPrompt(
        template=FEW_SHOT_RESPONSE["restaurant"],
        args_order=["history", "utterance", "state", "database"])
    
    expected_slots = ["pricerange", "area", "food", "name", "bookday", "bookpeople", "booktime"]



@dataclass
class FewShotHotelDefinition:
    state_prompt = FewShotPrompt(
        template=FEW_SHOT_STATE_PROMPT,
        args_order=["history", "utterance"],
        system_prompt=state_system_prompt("hotels"),
        sql_table=TABLES["hotel"]
    )
       
    response_prompt = FewShotPrompt(
        template=FEW_SHOT_RESPONSE["hotel"],
        args_order=["history", "utterance", "state", "database"])
    
    expected_slots = ["area", "internet", "parking", "stars", "type", "pricerange", "name", "bookday", "bookpeople", "bookstay"]




@dataclass
class FewShotTrainDefinition:
    state_prompt = FewShotPrompt(
        template=FEW_SHOT_STATE_PROMPT,
        args_order=["history", "utterance"],
        system_prompt=state_system_prompt("trains"),
        sql_table=TABLES["train"]
    )

    response_prompt = FewShotPrompt(
        template=FEW_SHOT_RESPONSE["train"],
        args_order=["history", "utterance", "state", "database"])

    expected_slots = ["arriveby", "leaveat", "bookpeople", "day", "departure", "destination"]


@dataclass
class FewShotTaxiDefinition:
    state_prompt = FewShotPrompt(
        template=FEW_SHOT_STATE_PROMPT,
        args_order=["history", "utterance"],
        system_prompt=state_system_prompt("taxis"),
        sql_table=TABLES["taxi"]
    )
       
    response_prompt = FewShotPrompt(
        template=FEW_SHOT_RESPONSE["taxi"],
    args_order=["history", "utterance", "state", "database"])

    expected_slots = ['departure', 'destination', 'leaveat', 'arriveby']


@dataclass
class FewShotAttractionDefinition:
    state_prompt = FewShotPrompt(
        template=FEW_SHOT_STATE_PROMPT,
        args_order=["history", "utterance"],
        system_prompt=state_system_prompt("attractions"),
        sql_table=TABLES["attraction"]
    )

    response_prompt = FewShotPrompt(
        template=FEW_SHOT_RESPONSE["attraction"],
        args_order=["history", "utterance", "state", "database"])
    
    expected_slots = ["type", "area", "name"]






SQL_FEW_SHOT_DOMAIN_DEFINITIONS = {
    "restaurant": FewShotRestaurantDefinition,
    "hotel": FewShotHotelDefinition,
    "train": FewShotTrainDefinition,
    "taxi": FewShotTaxiDefinition,
    "attraction": FewShotAttractionDefinition
}