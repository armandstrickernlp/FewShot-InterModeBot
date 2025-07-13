from dataclasses import dataclass
from prompts import FewShotPrompt, SimpleTemplatePrompt


# version: function call with call examples

FUNCTIONS = {
    "restaurant": {
            "name": "find_book_restaurant",
            "description": "Find a restaurant and book a table",
            "parameters": {
                "type": "object",
                "properties": {
                    "pricerange": {
                        "type": "string",
                        "description": "Price range of the restaurant",
                        "possible_values": ["cheap", "moderate", "expensive"],
                        "default_value": None
                    },
                    "area": {
                        "type": "string",
                        "description": "Area where the restaurant is located",
                        "possible_values": ["north", "east", "west", "south", "centre"],
                        "default_value": None
                    },
                    "food": {
                        "type": "string",
                        "description": "Type of food the restaurant serves",
                        "default_value": None
                    },
                    "name": {
                        "type": "string",
                        "description": "Name of the restaurant",
                        "default_value": None
                    },
                    "bookday": {
                        "type": "string",
                        "description": "Day of the booking",
                        "possible_values": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                        "default_value": None
                    },
                    "booktime": {
                        "type": "string",
                        "description": "Time of the booking. Map to a military format HH:MM",
                        "default_value": None
                    },
                    "bookpeople": {
                        "type": "integer",
                        "description": "Number of people for the booking",
                        "default_value": None
                    }
                }
            },
            "call_examples": [ # follow sql table examples
                "find_book_restaurant(pricerange=None, area=centre, food='italian', name='pizza hut city centre', bookday='wednesday', booktime='13:30', bookpeople=7)",
                "find_book_restaurant(pricerange='moderate', area='east', food=None, name='the missing sock', bookday=None, booktime=None, bookpeople=2)",
                "find_book_restaurant(pricerange='moderate', area='north', food='chinese', name='golden wok', bookday='friday', booktime='17:11', bookpeople=4)",
                "find_book_restaurant(pricerange=None, area='centre', food='british', name='cambridge chop house', bookday='monday', booktime='08:43', bookpeople=5)",
                "find_book_restaurant(pricerange='expensive', area='centre', food='modern european', name='darrys cookhouse and wine shop', bookday='saturday', booktime='11:20', bookpeople=8)"
            ]
        },
    "hotel": {
            "name": "find_book_hotel",
            "description": "Find a hotel and book a room",
            "parameters": {
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "description": "Area where the hotel is located",
                        "possible_values": ["north", "east", "west", "south", "centre"],
                        "default_value": None
                    },
                    "internet": {
                        "type": "boolean",
                        "description": "Internet availability",
                        "possible_values": ["yes", "no"],
                        "default_value": None
                    },
                    "parking": {
                        "type": "boolean",
                        "description": "Parking availability",
                        "possible_values": ["yes", "no"],
                        "default_value": None
                    },
                    "stars": {
                        "type": "integer",
                        "description": "Number of stars the hotel has",
                        "default_value": None
                    },
                    "type": {
                        "type": "string",
                        "description": "Type of the hotel",
                        "possible_values": ["hotel", "guesthouse"],
                        "default_value": None
                    },
                    "pricerange": {
                        "type": "string",
                        "description": "Price range of the hotel",
                        "possible_values": ["cheap", "moderate", "expensive"],
                        "default_value": None
                    },
                    "name": {
                        "type": "string",
                        "description": "Name of the hotel",
                        "default_value": None
                    },
                    "bookstay": {
                        "type": "integer",
                        "description": "Length of the stay",
                        "default_value": None
                    },
                    "bookday": {
                        "type": "string",
                        "description": "Day of the booking",
                        "possible_values": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                        "default_value": None
                    },
                    "bookpeople": {
                        "type": "integer",
                        "description": "Number of people for the booking",
                        "default_value": None
                    }
                }
            },
            "call_examples": [ # follow sql table examples
                "find_book_hotel(area='east', internet='yes', parking=None, stars=4, type='guesthouse', pricerange='moderate', name='a and b guest house', bookstay=3, bookday='friday', bookpeople=5)",
                "find_book_hotel(area='north', internet='yes', parking='yes', stars=5, type='hotel', pricerange='expensive', name='ashley hotel', bookstay=2, bookday='thursday', bookpeople=5)",
                "find_book_hotel(area='centre', internet='yes', parking='no', stars=None, type='guesthouse', pricerange='cheap', name='el shaddia guest house', bookstay=5, bookday='friday', bookpeople=2)",
                "find_book_hotel(area='east', internet='no', parking='yes', stars=None, type='guesthouse', pricerange=None, name='express by holiday inn cambridge', bookstay=3, bookday='monday', bookpeople=2)",
            ]
        },
    "train": {
            "name": "find_book_train",
            "description": "Find an available train and book a ticket",
            "parameters": {
                "type": "object",
                "properties": {
                    "arriveby": {
                        "type": "string",
                        "description": "Time the train should arrive at the latest. Map the value to a military format HH:MM",
                        "default_value": None
                    },
                    "leaveat": {
                        "type": "string",
                        "description": "Time the train should leave. Map the value to a military format HH:MM",
                        "default_value": None
                    },
                    "day": {
                        "type": "string",
                        "description": "Day the train should leave",
                        "possible_values": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                        "default_value": None
                    },
                    "departure": {
                        "type": "string",
                        "description": "Departure location. Should be a the name of a location.",
                        "default_value": None
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination location. Should be the name of a location.",
                        "default_value": None
                    },
                    "bookpeople": {
                        "type": "integer",
                        "description": "Number of people for the booking",
                        "default_value": None
                    }
                }
            },
            "call_examples": [ # follow sql table examples
                "find_book_train(arriveby='05:51', leaveat=None, day='monday', departure='london kings cross', destination='cambridge', bookpeople=6)",
                "find_book_train(arriveby='20:52', leaveat='20:24', day=None, departure='cambridge', destination='stansted airport', bookpeople=1)",
                "find_book_train(arriveby='12:56', leaveat='12:06', day='saturday', departure='peterborough', destination='cambridge', bookpeople=2)"
            ]
        },
    "taxi": {
            "name": "book_taxi",
            "description": "Book a taxi",
            "parameters": {
                "type": "object",
                "properties": {
                    "departure": {
                        "type": "string",
                        "description": "Departure location. Has to be the name of a location (can be restaurant/hotel/station/attraction...)",
                        "default_value": None
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination location. Has to be the name of a location (can be restaurant/hotel/station/attraction...)",
                        "default_value": None
                    },
                    "leaveat": {
                        "type": "string",
                        "description": "Time the taxi should leave. Map the value to a military format HH:MM",
                        "default_value": None
                    },
                    "arriveby": {
                        "type": "string",
                        "description": "Time the taxi should arrive at the latest. Map the value to a military format HH:MM",
                        "default_value": None
                    }
                }
            },
            "call_examples": [ 
                "book_taxi(departure='royal spice', destination='copper kettle', leaveat='14:45', arriveby='15:30')",
                "book_taxi(departure='university arms hotel', destination='magdalene college', leaveat=None, arriveby='15:45')",
                "book_taxi(departure='da vinci pizzeria', destination='lovell lodge', leaveat='11:45', arriveby=None)",
            ]
        },
    "hospital": {
            "name": "find_hospital",
            "description": "Find a hospital",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "description": "Department of the hospital",
                        "default_value": None
                    }
                }
            },
            "call_examples": [ 
                "find_hospital(department='maternity')",
                "find_hospital(department='accident and emergency')",
                "find_hospital(department='ophthalmology')",
            ]
        },
    "attraction": {
            "name": "find_attraction",
            "description": "Find an attraction",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Type of the attraction",
                        "possible_values": ["architecture", "sports", "entertainment", "cinema", "museum", "concert", "theatre", "park", "church", "hotel", "nightclub", "swimming pool", "college", "concert hall", "boat", "historical landmark", "gallery", "shopping area", "nature preserve", "sports venue", "theme park"],
                        "default_value": None
                    },
                    "area": {
                        "type": "string",
                        "description": "Area where the attraction is located",
                        "possible_values": ["north", "east", "west", "south", "centre"],
                        "default_value": None
                    },
                    "name": {
                        "type": "string",
                        "description": "Name of the attraction",
                        "default_value": None
                    }
                }
            },
            "call_examples": [
                "find_attraction(type='swimming pool', area='centre', name='abbey pool and astroturf pitch')",
                "find_attraction(type='theatre', area='centre', name='adc theatre')",
                "find_attraction(type='architecture', area=None, name='all saints church')",
                "find_attraction(type='museum', area='centre', name='castle galleries')",
            ]
        }
    }



# templates
FEW_SHOT_STATE_PROMPT = """
Output a function call with the correct function arguments given the customer's request. 
Make sure to follow the function definition.
Focus only on the values mentioned in the last utterance.
------
{}{}
---------
Now complete the following example:
Context: 
{}
Customer: {}
"""


ZERO_SHOT_STATE_PROMPT = """
Output a function call with the correct function arguments given the customer's request. 
Make sure to follow the function definition.
Focus only on the values mentioned in the last utterance.
------
Context: 
{}
Customer: {}
"""


def state_system_prompt(domain):
    return f"""
You are a task-oriented conversational AI assistant that helps users to book {domain}. 
Use the function definition below to create a function call with the correct arguments for the user's booking."""


FEW_SHOT_RESPONSE = {
    "restaurant": 
"""
Definition: You are an assistant that helps people to book a restaurant.
You can search for a restaurant by area, food, or pricerange.
There is also a number of restaurants in the database currently corresponding to the user's request.
If multiple restaurants are available, the Assistant should ask for further preferences. 
If you find a possible restaurant, the Assistant should provide [restaurant_name], [restaurant_address], [restaurant_phone] or [restaurant_postcode] if asked. Use these exact placeholders.
If no restaurants are available, the Assistant should ask for differrent preferences.
Before booking a table, the Assistant should ask for the time and the day of the booking and number of people. The Assistant should provide [reference] when the booking has been made. Use these exact placeholders.
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


# not upt to date (not used)
ZERO_SHOT_RESPONSE = {
    "restaurant": 
"""
Definition: You are an assistant that helps people to book a restaurant.
You can search for a restaurant by area, food, or pricerange.
There is also a number of restaurants in the database currently corresponding to the user's request.
If you find a possible restaurant, provide [restaurant_name], [restaurant_address], [restaurant_phone] or [restaurant_postcode] if asked. Use these exact placeholders.
If booking a table, provide [ref] in the answer. Use these exact placeholders.
Always act as if the booking is successfully done.
Write the Assistant response as a single line, based on the state and the database.
------
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
If you find a hotel, provide [hotel_name], [hotel_address], [hotel_phone] or [hotel_postcode] if asked. Use these exact placeholders.
If booking a hotel, provide [reference] in the answer. Use these exact placeholders.
Always act as if the booking is successfully done.
Write the Assistant response as a single line, based on the state and the database.
------
input:{}
Customer: {}
state: {}
database: {}
Assistant:""",

    "train":
"""
Definition: You are an assistant that helps people to find a train connection.
The customer needs to specify the departure and destination station, and the time of departure or arrival.
There is also a number of trains in the database currently corresponding to the user's request.
If you find a train, provide [arriveby], [leaveat] or [departure] if asked. Use these exact placeholders.
If booking, provide [reference] in the answer. Use these exact placeholders.
Always act as if the booking is successfully done.
Write the Assistant response as a single line, based on the state and the database.
------
input:{}
Customer: {}
state: {}
database: {}
Assistant:""",
    
        "taxi":
"""
Definition: You are an assistant that helps people to book a taxi.
Provide the type of the car as [type] and [phone] as the phone number in the answer. Use these exact placeholders.
Always act as if the booking is successfully done.
Write the Assistant response as a single line, based on the state.
------
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
If you find an attraction, provide [attraction_name], [attraction_address], [attraction_phone] or [attraction_postcode] if asked. Use these exact placeholders. 
Write the Assistant's response as a single line, based on the state and the database.
------
input:{}
Customer: {}
state: {}
database: {}
Assistant:""",

    "hospital":
"""
Definition: You are an assistant that helps people to find a hospital. 
Write the Assistant response as a single line, based on the state.
------
input:{}
Customer: {}
state: {}
database: {}
Assistant:""",
}

###############
# Few-shot definitions
###############

@dataclass
class FewShotRestaurantDefinition:
    state_prompt = FewShotPrompt(
        template=FEW_SHOT_STATE_PROMPT,
        args_order=["history", "utterance"],
        system_prompt= state_system_prompt("restaurants"),
        function_definition=FUNCTIONS['restaurant']
    )    
    response_prompt = FewShotPrompt(
        template=FEW_SHOT_RESPONSE['restaurant'],
        args_order=["history", "utterance", "state", "database"])
    
    expected_slots = ["pricerange", "area", "food", "name", "bookday", "bookpeople", "booktime"]


@dataclass
class FewShotHotelDefinition:
    state_prompt = FewShotPrompt(
        template = FEW_SHOT_STATE_PROMPT,
        args_order = ["history", "utterance"],
        system_prompt = state_system_prompt("hotels"),
        function_definition = FUNCTIONS['hotel']
    )
       
    response_prompt = FewShotPrompt(
        template=FEW_SHOT_RESPONSE['hotel'],
        args_order=["history", "utterance", "state", "database"])
    
    expected_slots = ["area", "internet", "parking", "stars", "type", "pricerange", "name", "bookday", "bookpeople", "bookstay"]


@dataclass
class FewShotTrainDefinition:
    state_prompt = FewShotPrompt(
        template=FEW_SHOT_STATE_PROMPT,
        args_order=["history", "utterance"],
        system_prompt=state_system_prompt("trains"),
        function_definition=FUNCTIONS['train']
    )

    response_prompt = FewShotPrompt(
        template=FEW_SHOT_RESPONSE['train'],
        args_order=["history", "utterance", "state", "database"])

    expected_slots = ["arriveby", "leaveat", "bookpeople", "day", "departure", "destination"]


@dataclass
class FewShotTaxiDefinition:
    state_prompt = FewShotPrompt(
        template=FEW_SHOT_STATE_PROMPT,
        args_order=["history", "utterance"],
        system_prompt=state_system_prompt("taxis"),
        function_definition=FUNCTIONS['taxi']
    )
       

    response_prompt = FewShotPrompt(
        template=FEW_SHOT_RESPONSE['taxi'],
        args_order=["history", "utterance", "state", "database"])

    expected_slots = ['departure', 'destination', 'leaveat', 'arriveby']


@dataclass
class FewShotHospitalDefinition:
    state_prompt = FewShotPrompt(
        template=FEW_SHOT_STATE_PROMPT, 
        args_order=["history", "utterance"],
        system_prompt=state_system_prompt("hospitals"),
        function_definition=FUNCTIONS['hospital']
    )
        
    response_prompt = FewShotPrompt(
        template=FEW_SHOT_RESPONSE['hospital'],
        args_order=["history", "utterance", "state", "database"])
    
    expected_slots = ['department']


@dataclass
class FewShotAttractionDefinition:
    state_prompt = FewShotPrompt(
        template=FEW_SHOT_STATE_PROMPT,
        args_order=["history", "utterance"],
        system_prompt=state_system_prompt("attractions"),
        function_definition=FUNCTIONS['attraction']
    )

    response_prompt = FewShotPrompt(
        template=FEW_SHOT_RESPONSE['attraction'],
        args_order=["history", "utterance", "state", "database"])
    
    expected_slots = ["type", "area", "name"]



#################
# Zero-shot definitions
#################
# response prompt is not used in the zero-shot setting

# à revoir
@dataclass
class ZeroShotRestaurantDefinition:
    state_prompt = SimpleTemplatePrompt(
        template=ZERO_SHOT_STATE_PROMPT,
        args_order=["history", "utterance"],
        system_prompt=state_system_prompt("restaurants"),
        function_definition=FUNCTIONS['restaurant']
    )


    response_prompt = SimpleTemplatePrompt(
        template=ZERO_SHOT_RESPONSE['restaurant'],
        args_order=["history", "utterance", "state", "database"])
    
    expected_slots = ["pricerange", "area", "food", "name", "bookday", "bookpeople", "booktime"]


@dataclass
class ZeroShotHotelDefinition:
    state_prompt = SimpleTemplatePrompt(
        template=ZERO_SHOT_STATE_PROMPT,
        args_order=["history", "utterance"],
        system_prompt=state_system_prompt("hotels"),
        function_definition=FUNCTIONS['hotel']
    )
        
    response_prompt = SimpleTemplatePrompt(
        template=ZERO_SHOT_RESPONSE['hotel'],
        args_order=["history", "utterance", "state", "database"])
    
    expected_slots = ["area", "internet", "parking", "stars", "type", "pricerange", "name", "booktime", "bookpeople", "bookstay"]


@dataclass
class ZeroShotTrainDefinition:
    state_prompt = SimpleTemplatePrompt(
        template=ZERO_SHOT_STATE_PROMPT,
        args_order=["history", "utterance"],
        system_prompt=state_system_prompt("trains"),
        function_definition=FUNCTIONS['train']
    )

    response_prompt = SimpleTemplatePrompt(
        template=ZERO_SHOT_RESPONSE['train'],
        args_order=["history", "utterance", "state", "database"])

    expected_slots = ["arriveby", "leaveat", "bookpeople", "day", "departure", "destination"]


@dataclass
class ZeroShotTaxiDefinition:
    state_prompt = SimpleTemplatePrompt(
        template=ZERO_SHOT_STATE_PROMPT,
        args_order=["history", "utterance"],
        system_prompt=state_system_prompt("taxis"),
        function_definition=FUNCTIONS['taxi']
    )

    response_prompt = SimpleTemplatePrompt(
        template=ZERO_SHOT_RESPONSE['taxi'],
    args_order=["history", "utterance", "state", "database"])

    expected_slots = ['departure', 'destination', 'leaveat', 'arriveby']


@dataclass
class ZeroShotHospitalDefinition:
    state_prompt = SimpleTemplatePrompt(
        template=ZERO_SHOT_STATE_PROMPT,
        args_order=["history", "utterance"],
        system_prompt=state_system_prompt("hospitals"),
        function_definition=FUNCTIONS['hospital']
    )

    response_prompt = SimpleTemplatePrompt(
        template=ZERO_SHOT_RESPONSE['hospital'],
        args_order=["history", "utterance", "state", "database"])
    
    expected_slots = ['department']


@dataclass
class ZeroShotAttractionDefinition:
    state_prompt = SimpleTemplatePrompt(
        template=ZERO_SHOT_STATE_PROMPT,
        args_order=["history", "utterance"],
        system_prompt=state_system_prompt("attractions"),
        function_definition=FUNCTIONS['attraction']
    )
           

    response_prompt = SimpleTemplatePrompt(
        template=ZERO_SHOT_RESPONSE['attraction'],
        args_order=["history", "utterance", "state", "database"])
    expected_slots = ["type", "area", "name"]






DOMAIN_2_FUNCTION= {
    "train": "find_book_train",
    "restaurant": "find_book_restaurant",
    "hotel": "find_book_hotel",
    "taxi": "book_taxi",
    "attraction": "find_attraction",
    "hospital": "find_hospital"
}

FUNCTION_2_DOMAIN = {
    "find_book_train": "train",
    "find_book_restaurant": "restaurant",
    "find_book_hotel": "hotel",
    "book_taxi": "taxi",
    "find_attraction": "attraction",
    "find_hospital": "hospital"
}


FC_FEW_SHOT_DOMAIN_DEFINITIONS = {
    "restaurant": FewShotRestaurantDefinition,
    "hotel": FewShotHotelDefinition,
    "train": FewShotTrainDefinition,
    "taxi": FewShotTaxiDefinition,
    "attraction": FewShotAttractionDefinition,
    "hospital": FewShotHospitalDefinition
}

FC_ZERO_SHOT_DOMAIN_DEFINITIONS = {
    "restaurant": ZeroShotRestaurantDefinition,
    "hotel": ZeroShotHotelDefinition,
    "train": ZeroShotTrainDefinition,
    "taxi": ZeroShotTaxiDefinition,
    "attraction": ZeroShotAttractionDefinition,
    "hospital": ZeroShotHospitalDefinition
}

