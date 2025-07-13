from prompts import SimpleTemplatePrompt


domain_selection_prompt = SimpleTemplatePrompt(
      template = """
Select the appropriate domain given the customer's request. 
Respond with only one word: the domain name.
It is very important you focus on the customer's current request to make your decision.
Context: 
{}
{}
Domain:""",
        args_order=["history", "utterance"],
        system_prompt = """
You are a conversational AI, capable of detecting the domain of a user request.
The only possible domains are :
- train (booking train tickets)
- restaurant (finding and booking restaurants)
- hotel (finding and booking hotels)
- attraction (eg. "architecture", "sports", "entertainment", "cinema", "museum", "concert", "theatre", "park", "church", "hotel", "nightclub", "swimming pool", "college", "concert hall", "boat", "historical landmark", "gallery", "shopping area", "nature preserve", "sports venue", "theme park")
- taxi (booking a taxi from one location to another)
Return only the domain name."""
)

task_selection_prompt = SimpleTemplatePrompt(
    template="""
Use the Dialogue Context and the User Turn to classifiy the User Turn as "chitchat" or "task". 
The criteria are:
If the User Turn contains:
    - comments about personal life, opinions, or experiences 
    - casual comments about Cambridge or the domains (restaurants, trains, hotels, taxis, attractions) 
then the User Turn is "chitchat".

If the User Turn contains 
    - an *explicit* request for information in a task domain (restaurant, train, hotel, attraction, taxi) or 
    - a request to perform an action in a task domain (restaurant, train, hotel, attraction, taxi) or 
    - an essential piece of information relevant to a task domain (restaurant, train, hotel, attraction, taxi)
then the User Turn is "task".

Dialogue Context:
{}
User Turn:
{}

Respond with one word. Either "chitchat" or "task" between <label> and </label> tags.
Label:""",
    args_order=["history", "utterance"],
    system_prompt = """A user is using Cambridge's Towninfo Centre information assistant which can help users with information and bookings. These are tasks with certain specific domains. 
    Task domains include:
- train (booking train tickets)
- restaurant (finding and booking restaurants)
- hotel (finding and booking hotels)
- attraction (eg. "architecture", "sports", "entertainment", "cinema", "museum", "concert", "theatre”...)
- taxi (booking a taxi from one location to another)

You are an expert at determining if a User Turn contains task-related information or requests."""
)
