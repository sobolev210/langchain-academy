import requests
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


def slang_search(query: str) -> str:
    """Search a word or phrase in slang dictionary. Takes subject of search string as input"""
    try:
        response = requests.get(
            "https://unofficialurbandictionaryapi.com/api/search",
            params={"term": query, "limit": 3},
        )
        response.raise_for_status()
        data = response.json()

        if not data or "data" not in data or not data["data"]:
            return f"No definition found for '{query}'"

        definitions = []
        for i, entry in enumerate(data["data"][:3], 1):
            meaning = entry.get("meaning", "").replace("[", "").replace("]", "")
            example = entry.get("example", "").replace("[", "").replace("]", "")
            definitions.append(
                f"{i}. {meaning}" + (f"\n   Example: {example}" if example else "")
            )

        return f"Definitions for '{query}':\n" + "\n\n".join(definitions)
    except Exception as e:
        return f"Error searching for '{query}': {str(e)}"


llm = ChatOpenAI(model="gpt-5-mini")
llm_with_tools = llm.bind_tools([slang_search])

sys_msg = SystemMessage(
    content="You are a helpful assistant that can search helps to explain what words means, especially slang ones. "
    "Double check definition with slang search before giving any response."
)
sys_msg_slangy_llm = SystemMessage(
    "Rewrite definition in one sentence, and give small rap example with usage of this word (quatrain form)"
)


def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


def definition_rap_assistant(state: MessagesState):
    return {"messages": [llm.invoke([sys_msg_slangy_llm] + state["messages"])]}


memory = MemorySaver()
builder = StateGraph(state_schema=MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode([slang_search]))
builder.add_node("definition_rap_assistant", definition_rap_assistant)
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    source="assistant",
    path=tools_condition,
    path_map={"tools": "tools", "__end__": "definition_rap_assistant"},
)
builder.add_edge("tools", "assistant")
builder.add_edge("definition_rap_assistant", END)

graph = builder.compile(checkpointer=memory)

# graph.invoke({"messages": [HumanMessage(content="What is the meaning of 'slang'?")]})
