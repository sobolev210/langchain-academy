import json
import sqlite3

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from typing_extensions import TypedDict

db_path = "/Users/andreisobolev/Studying/Langchain-course/langchain-academy/module-2/example_module2.db"
conn = sqlite3.connect(db_path, check_same_thread=False)

load_dotenv()


class InputState(MessagesState):
    """Only messages are accepted as input"""

    pass


class OutputState(TypedDict):
    answer: str


class OverallState(MessagesState):
    """Internal state that includes definitions tracking"""

    definitions: dict[str, str] | None = None


general_agent_instruction = SystemMessage(
    """You're general purpose assistant. Answer user's questions. Answer shortly"""
)

defintions_extraction_instruction = """
    "You're maintaining a dict of definitions that user was asking about."
    "You're given existing definitions dict, and last messages in current conversation. If user was asking about definition of smth, or clarifications for definitions, "
    "add new / extend existing one to this dict, and return just the key-values that needs to be updates. Return json, keep it empty if no changes needed. Current dict: "
"""

llm = ChatOpenAI(model="gpt-5-mini")
# memory = MemorySaver()
memory = SqliteSaver(conn)


def general_assistant_node(state: InputState) -> InputState:
    return {"messages": [llm.invoke([general_agent_instruction] + state["messages"])]}


def defintions_assistant(state: OverallState) -> OutputState:
    try:
        current_definitions = state.get("definitions")
        system_message = SystemMessage(
            content=defintions_extraction_instruction
            + (json.dumps(current_definitions) if current_definitions else "")
        )
        response = llm.invoke([system_message] + state["messages"][-2:])
        # TODO try to use reducer function
        new_defintions = json.loads(response.content)
        if current_definitions:
            current_definitions.update(new_defintions)
        else:
            current_definitions = new_defintions
    except Exception as e:
        print(e)
        raise e
    return {"definitions": current_definitions, "answer": state["messages"][-1]}


config = {"configurable": {"thread_id": "1"}}


workflow = StateGraph(
    OverallState,
    input_schema=InputState,
    output_schema=OutputState,
)
workflow.add_node("general_assistant", general_assistant_node)
workflow.add_node("definitions_maintainer", defintions_assistant)
workflow.add_edge(START, "general_assistant")
workflow.add_edge("general_assistant", "definitions_maintainer")
workflow.add_edge("definitions_maintainer", END)

graph = workflow.compile(checkpointer=memory)

# graph.invoke({"messages": HumanMessage("Hey how are you?"), "definitions": {}})
# graph.invoke({"messages": HumanMessage("Hey how are you?"), "definitions": {}})

config = {"configurable": {"thread_id": "4"}}

graph.invoke({"messages": HumanMessage("What's KPI?")}, config)
graph.invoke({"messages": HumanMessage("What's pennyboard?")}, config)
# graph.invoke({"messages": HumanMessage("What's AGI?")}, config)
