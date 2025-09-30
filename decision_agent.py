from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain


class WaaMAgent:
    def __init__(self, name, profile, llm, use_memory=True):
        self.name = name
        self.profile = profile
        self.llm = llm
        self.use_memory = use_memory

        if self.use_memory:
            self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        else:
            self.memory = None  # No memory used

        # Define output parser
        self.parser = StructuredOutputParser.from_response_schemas([
            ResponseSchema(
                name="answer",
                description="An integer value between 1 and 9 inclusive, representing the preference strength"
            ),
            ResponseSchema(
                name="agent_reasoning",
                description="A short, concise explanation (1–3 sentences) justifying the chosen value"
            )
        ])

        # Define prompt template
        self.prompt_template = PromptTemplate(
            input_variables=[
                "main_question", "instructions", "factor_1", "factor_2",
                "description", "style", "priorities", "format_instructions"
            ],
            template="""
                        You are a {description}
                        Your communication style is: {style}
                        Your top decision priorities are: {priorities}

                        Main Question: {main_question}
                        Instructions: {instructions}

                        Now compare the following two factors:
                        - {factor_1}
                        - {factor_2}

                        Choose a number between **1 and 9**:
                        - 1–4 → {factor_1} is more important
                        - 5 → Equal importance
                        - 6–9 → {factor_2} is more important

                        {format_instructions}
                    """
        )

    def evaluate(self, factor_1, factor_2, main_question, instructions):
        try:
            prompt = self.prompt_template.format(
                factor_1=factor_1,
                factor_2=factor_2,
                main_question=main_question,
                instructions=instructions.replace("<br>", " ").replace("<br/>", " "),
                description=self.profile["description"],
                style=self.profile["style"],
                priorities=", ".join(self.profile["priorities"]),
                format_instructions=self.parser.get_format_instructions()
            )

            response = self.llm.invoke(prompt).strip()
            return self.parser.parse(response)

        except Exception as e:
            print(f"⚠️ {self.name} failed: {e}")
            return {
                "answer": 5,
                "agent_reasoning": "Fallback: Equal importance due to model failure."
            }

    def reset_memory(self):
        if self.memory:
            self.memory.clear()
