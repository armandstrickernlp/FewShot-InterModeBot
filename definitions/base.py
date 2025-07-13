from dataclasses import dataclass
from prompts import FewShotPrompt, SimpleTemplatePrompt


# detect domain
multiwoz_domain_prompt = SimpleTemplatePrompt(template="""
Determine which domain is considered in the following dialogue situation.
Choose one domain from this list:
 - restaurant
 - hotel
 - attraction
 - taxi
 - train
Answer with only one word, the selected domain from the list.
You have to always select the closest possible domain.
Consider the last domain mentioned, so focus mainly on the last utterance.

-------------------
Example1:
Customer: I need a cheap place to eat
Assistant: We have several not expensive places available. What food are you interested in?
Customer: Chinese food.

Domain: restaurant

-------

Example 2:
Customer: I also need a hotel in the north.
Assistant: Ok, can I offer you the Molly's place?
Customer: What is the address?

Domain: hotel

---------

Example 3:
Customer: What is the address?
Assistant: It's 123 Northfolk Road.
Customer: That's all. I also need a train from London.

Domain: train
""

Now complete the following example:
{}
{}
Domain:""", args_order=["history", "utterance"])

"""
######################
FEW SHOT
######################
"""



FEW_SHOT_RESPONSE = {
    "restaurant": 
"""
Definition: You are an assistant that helps people to book a restaurant.
You can search for a restaurant by area, food, or pricerange.
There is also a number of restaurants in the database currently corresponding to the user's request.
If multiple restaurants are available, the Assistant should ask for further preferences. 
If you find a possible restaurant, the Assistant should provide [restaurant_name], [restaurant_address], [restaurant_phone] or [restaurant_postcode] if asked. Use these exact placeholders.
If no restaurants are available, the Assistant should ask for differrent preferences.
If booking a table, the Assistant should provide [reference] in the answer. Use these exact placeholders.
**Always act as if booking is available.**
Write the Assistant response as a single line, based on the state and the database.
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
If booking a hotel, the Assistant should provide [ref] in the answer. Use these exact placeholders.
**Always act as if booking is available.**
Write the Assistant response as a single line, based on the state and the database.
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
The customer needs to specify the departure and destination station, and the time of departure or arrival.
There is also a number of trains in the database currently corresponding to the user's request.
If you find a train, the Assistant should provide [arriveby], [leaveat], [departure] or [duration] if asked. Use these exact placeholders.
If booking, the Assistant should provide [ref] in the answer. Use these exact placeholders.
**Always act as if booking is available.**
Write the Assistant response as a single line, based on the state and the database.
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
Write the Assistant response as a single line, based on the state.
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
Write the Assistant's response as a single line, based on the state and the database.
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
Write the Assistant response as a single line, based on the state.
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
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - "pricerange" that specifies the price range of the restaurant (cheap/moderate/expensive)
 - "area" that specifies the area where the restaurant is located (north/east/west/south/centre)
 - "food" that specifies the type of food the restaurant serves
 - "name" that specifies the name of the restaurant
 - "bookday" that specifies the day of the booking
 - "booktime" that specifies the time of the booking
 - "bookpeople" that specifies for how many people is the booking made
Do not capture any other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template=FEW_SHOT_RESPONSE["restaurant"],
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["pricerange", "area", "food", "name", "bookday", "bookpeople", "booktime"]

@dataclass
class FewShotHotelDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.

Values that should be captured are:
 - "area" that specifies the area where the hotel is located (north/east/west/south/centre)
 - "internet" that specifies if the hotel has internet (yes/no)
 - "parking" that specifies if the hotel has parking (yes/no)
 - "stars" that specifies the number of stars the hotel has (1/2/3/4/5)
 - "type" that specifies the type of the hotel (hotel/bed and breakfast/guest house)
 - "pricerange" that specifies the price range of the hotel (cheap/expensive)
 - "name" that specifies name of the hotel
 - "bookstay" specifies length of the stay
 - "bookday" specifies the day of the booking
 - "bookpeople" specifies how many people should be booked for.
Do not capture any other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template=FEW_SHOT_RESPONSE["hotel"],
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["area", "internet", "parking", "stars", "type", "pricerange", "name", "bookday", "bookpeople", "bookstay"]


@dataclass
class FewShotTrainDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.

Values that should be captured are:
 - "arriveby" that specifies what time the train should arrive
 - "leaveat" that specifies what time the train should leave
 - "day" that specifies what day the train should leave (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
 - "departure" that specifies the departure station
 - "destination" that specifies the destination station
 - "bookpeople" that specifies how many people the booking is for
Do not capture any other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template=FEW_SHOT_RESPONSE["train"],
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["arriveby", "leaveat", "bookpeople", "day", "departure", "destination"]


@dataclass
class FewShotTaxiDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.

If not specified, leave the value empty.
Values that should be captured are:
 - "arriveby" that specifies what time the train should arrive
 - "leaveat" that specifies what time the train should leave
 - "departure" that specifies the departure station
 - "destination" that specifies the destination station
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template=FEW_SHOT_RESPONSE["taxi"],
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ['departure', 'destination', 'leaveat', 'arriveby']



@dataclass
class FewShotHospitalDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.

 - "department" that specifies the department of interest
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template=FEW_SHOT_RESPONSE["hospital"],
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ['department']



@dataclass
class FewShotBusDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.

If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template="""
Definition: You are an assistant that helps people to find a bus.
------
{}{}
---------
Now complete the following example:
input:{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = []


@dataclass
class FewShotAttractionDefinition:
    state_prompt = FewShotPrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.

Values that should be captured are:
 - "type" that specifies the type of attraction (museum/gallery/theatre/concert/stadium)
 - "area" that specifies the area where the attraction is located (north/east/west/south/centre)
 - "name" that specigies the name of the attraction
Do not capture any other values!
If not specified, leave the value empty.
------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template=FEW_SHOT_RESPONSE["attraction"],
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["type", "area", "name"]


"""
######################
ZERO SHOT
######################
"""


@dataclass
class ZeroShotRestaurantDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - "pricerange" that specifies the price range of the restaurant (cheap/moderate/expensive)
 - "area" that specifies the area where the restaurant is located (north/east/west/south/centre)
 - "food" that specifies the type of food the restaurant serves
 - "name" that is the name of the restaurant
Do not capture any other values!
If not specified, leave the value empty.
input: {}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
Definition: You are an assistant that helps people to book a restaurant.
You can search for a restaurant by area, food, or price.
There is also a number of restaurants in the database currently corresponding to the user's request.
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [address].
If you find a restaurant, provide [restaurant_name], [restaurant_address], [restaurant_phone] or [restaurant_postcode] if asked.
If booking, provide [reference] in the answer.
input:{}
Customer: {}
state: {}
database: {}
output:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["pricerange", "area", "food", "name"]


@dataclass
class ZeroShotHotelDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - "area" that specifies the area where the hotel is located (north/east/west/south/centre)
 - "internet" that specifies if the hotel has internet (yes/no)
 - "parking" that specifies if the hotel has parking (yes/no)
 - "stars" that specifies the number of stars the hotel has (1/2/3/4/5)
 - "type" that specifies the type of the hotel (hotel/bed and breakfast/guest house)
 - "pricerange" that specifies the price range of the hotel (cheap/expensive)
Do not capture any other values!
If not specified, leave the value empty.
input: {}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
    Definition: You are an assistant that helps people to book a hotel.
The customer can ask for a hotel by name, area, parking, internet availability, or price.
There is also a number of hotel in the database currently corresponding to the user's request.
If you find a hotel, provide [hotel_name], [hotel_address], [hotel_phone] or [hotel_postcode] if asked.
Do not provide real entities in the response! Just provide entity name in brackets, like [name] or [address].
If booking, provide [reference] in the answer.
input:{}
Customer: {}
state: {}
database: {}
output:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["area", "internet", "parking", "stars", "type", "pricerange"]


@dataclass
class ZeroShotTrainDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - "arriveBy" that specifies what time the train should arrive
 - "leaveAt" that specifies what time the train should leave
 - "day" that specifies what day the train should leave (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
 - "departure" that specifies the departure station
 - "destination" that specifies the destination station
Do not capture any other values!
If not specified, leave the value empty
input: {}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
Definition: You are an assistant that helps people to find a train connection.
The customer needs to specify the departure and destination station, and the time of departure or arrival.
There is also a number of trains in the database currently corresponding to the user's request.
If you find a train, provide [arriveby], [leaveat] or [departure] if asked.
Do not provide real entities in the response! Just provide entity name in brackets, like [duration] or [price].
If booking, provide [reference] in the answer.
input:{}
Customer: {}
state: {}
database: {}
output:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["arriveBy", "leaveAt", "day", "departure", "destination"]


@dataclass
class ZeroShotTaxiDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
If not specified, leave the value empty.
Values that should be captured are:
 - "arriveBy" that specifies what time the train should arrive
 - "leaveAt" that specifies what time the train should leave
 - "departure" that specifies the departure station
 - "destination" that specifies the destination station
 - "day" that specifies what day the train should leave (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
input: {}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
Definition: You are an assistant that helps people to book a taxi.
Do not provide real entities in the response! Just provide entity name in brackets, like [color] or [type].
input:{}
Customer: {}
state: {}
database: {}
output:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ['departure', 'destination', 'leaveAt', 'arriveBy', 'date']



@dataclass
class ZeroShotHospitalDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
If not specified, leave the value empty.
input: {}
Customer: {}
output:
state:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
Definition: You are an assistant that helps people to find a hospital.
input:{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = []



@dataclass
class ZeroShotBusDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
If not specified, leave the value empty.
input: {}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
Definition: You are an assistant that helps people to find a bus.
input:{}
Customer: {}
state: {}
database: {}
output:response:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = []


@dataclass
class ZeroShotAttractionDefinition:
    state_prompt = SimpleTemplatePrompt(template="""
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - "type" that specifies the type of attraction (museum/gallery/theatre/concert/stadium)
 - "area" that specifies the area where the attraction is located (north/east/west/south/centre)
Do not capture ay other values!
If not specified, leave the value empty.
input: {}
Customer: {}
output:""",
                                    args_order=["history", "utterance"])
    response_prompt = SimpleTemplatePrompt(template="""
    Definition: You are an assistant that helps people to find an attraction.
The customer can ask for an attraction by name, area, or type.
There is also a number of restaurants provided in the database.
Do not provide real entities in the response! Just provide entity name in brackets, like [address] or [name].
If you find a hotel, provide [attraction_name], [attraction_address], [attraction_phone] or [attraction_postcode] if asked.
input:{}
Customer: {}
state: {}
database: {}
output:""",
                                args_order=["history", "utterance", "state", "database"])
    expected_slots = ["type", "area"]


MW_FEW_SHOT_DOMAIN_DEFINITIONS = {
    "restaurant": FewShotRestaurantDefinition,
    "hotel": FewShotHotelDefinition,
    "attraction": FewShotAttractionDefinition,
    "train": FewShotTrainDefinition,
    "taxi": FewShotTaxiDefinition,
    "hospital": FewShotHospitalDefinition, 
}


MW_ZERO_SHOT_DOMAIN_DEFINITIONS = {
    "restaurant": ZeroShotRestaurantDefinition,
    "hotel": ZeroShotHotelDefinition,
    "attraction": ZeroShotAttractionDefinition,
    "train": ZeroShotTrainDefinition,
    "taxi": ZeroShotTaxiDefinition,
    "hospital": ZeroShotHospitalDefinition,
}



