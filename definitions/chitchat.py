from prompts import SimpleTemplatePrompt

chitchat_prompt = SimpleTemplatePrompt(
    template="""Respond to the user with a short response. Focus on being friendly and engaging. Write the resppnse in between <response> and </response> tags.
    Context:
    {}
    {}
    Response:""",
    args_order=["history", "utterance"],
    system_prompt="""You are a friendly conversational AI. Your goal is to engage with the user in a friendly conversation."""
)


