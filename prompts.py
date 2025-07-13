from typing import Any, Text, Dict, List
from dataclasses import dataclass

@dataclass
class SimpleTemplatePrompt:
    template: str
    args_order: list
    system_prompt: str = None
    function_definition: Dict[str, str] = None
    sql_table: str = None
    
    def __call__(self, **kwargs: Any) -> Text:
        args = [kwargs[arg] for arg in self.args_order]
        return self.template.format(*args), self.system_prompt, self.function_definition, self.sql_table


@dataclass
class FewShotPrompt(SimpleTemplatePrompt):

    def __call__(self, positive_examples: list[Dict], negative_examples: list[Dict], **kwargs) -> Any:
        positive_examples = self._process_positive_examples(positive_examples)
        negative_examples = self._process_negative_examples(negative_examples)
        args = [kwargs[arg] for arg in self.args_order]
        return self.template.format(positive_examples, negative_examples, *args), self.system_prompt, self.function_definition, self.sql_table
    
    def _process_positive_examples(self, positive_examples: list) -> Text:
        output = "\n"
        for n, example in enumerate(positive_examples):
            output += "---------------------" + \
                      f"Example {n}:\n" + \
                      f"{example['input']}\n" + \
                      f"\n{example['output']}\n"
        return output + "\n"
    
    def _process_negative_examples(self, negative_examples: list) -> Text:
        output = "\n"
        for n, example in enumerate(negative_examples):
            output += f"Negative example {n}:\n" + \
                      f"{example['input']}\n" + \
                      f"\n{example['output']}\n"
        return output + "\n"

