"""
conferences
hackathons
intensives


"""

import operator
import sqlite3
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore
from langgraph.types import Send
from psycopg import Connection
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(model="gpt-5-mini")


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")


class ResultsQualityAssesment(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the results are sufficient or need to continue searching."
    )
    required_improvements: str = Field(
        description="If the results are not sufficient, what are the required improvements (i.e. what's not fully correct or might be missing or might be improved) comparing to current documents."
    )


query_generation_instruction = SystemMessage(
    content="""You will be given a type of opportunity in ML field that AI assistant is searching for.

    Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.

    Pay particular attention to type of opportunity and additional context from previous messages

    Create well structured search query.
    Keep it concise: **max 200 characters.**
    """
)

search_instructions = SystemMessage(
    """
    You're an assistant asked to help with searching different opportunities in Machine learning field. You will be given a type of opportunity you need to search for, e.g. hackathons. Additionally, you will be given a time constraint - within which period to seach for such opportunities, e..g within next 3 months, and other information such as in which location to search if applicable.
    Your end goal is to help the user to find the best opportunities to develop and grow in Machine Learning field (gen AI, traditional machine learning).
    Now, given the type of opportunity to search, make the search query.
    Opportunity type: {opportunity_type}
    Additional context: {additional_context}
    Do not ask for any additional information, just make the search query.
    Ensure you're looking for future events only, not past ones.
    Additionally, here're proposed refinements from previous search results, incorporate them into your search query: {required_improvements} and previous search query: {previous_search_query}. If empty, just ignore them
    """
)

generate_report_instructions = SystemMessage(
    """
You're given search results for specific type of opportunities in ML field.
Your task is to compose structured report based on found documents.
Provide key information, such as dates, location, duration, topic, cost, or any other information you find useful. The set of attributes is flexible.

Provide final result in text format, eg.
--------
Event 1
Topic: ...
Short summary: ...
Date: ...
...
Link: ...
--------
Event 2
Topic: ...
Short summary: ...
Date: ...
...
Link: ...

Now, here's the list of documents found:
{found_documents}
If no documents, just say "No documents found for type of opportunity: {opportunity_type}"
"""
)

assess_quality_instruction = """ 
You're an assistant asked to help with searching different opportunities in Machine learning field. Your task is to evaluate the quality of found opportunities.
You're given search results for opportunity type: {opportunity_type} and with additional search criteria provided by user {additional_context}
Also, you're given search results in summarized format.
Your task is to say if what's given you now is good enough or requires improvements.
If the result is not good enough, say what're the current issues and what could be improved.

Search results summary:
{search_results_summarized}

If summary says no documents, just assume it's enough, as we might not get better results in next iterations.
"""


class ResearcherInstanceInternalState(MessagesState):
    opportunity_type: str
    additional_context: str
    last_assessment_results: ResultsQualityAssesment
    previous_search_query: SearchQuery
    found_documents: Annotated[list, operator.add]
    researcher_reports: list[str]
    iterations_count: int = 0


def search_web(state: ResearcherInstanceInternalState):
    """Retrieve docs from web search"""

    # Search
    tavily_search = TavilySearch(max_results=3, include_raw_content=True)

    last_assessment_results = state.get("last_assessment_results")
    previous_search_query = state.get("previous_search_query")

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_instruction = SystemMessage(
        search_instructions.content.format(
            opportunity_type=state["opportunity_type"],
            additional_context=state["additional_context"],
            required_improvements=last_assessment_results.required_improvements
            if last_assessment_results
            else "",
            previous_search_query=previous_search_query.search_query
            if previous_search_query
            else "",
        )
    )
    search_query: SearchQuery = structured_llm.invoke(
        [search_instruction] + state["messages"]
    )
    raw_query = (search_query.search_query or "").strip()
    if len(raw_query) > 380:
        raw_query = raw_query[:380]

    # Search
    data = tavily_search.invoke({"query": raw_query})
    if not isinstance(data, dict):
        return {"found_documents": [""]}
    search_docs = data.get("results", data)
    if not isinstance(search_docs, list):
        search_docs = [search_docs]

    # Format
    formatted_documents = []
    for doc in search_docs:
        if isinstance(doc, str):
            formatted_documents.append(f"<Document>\n{doc}\n</Document>")
            continue
        if isinstance(doc, dict):
            title = doc.get("title") or ""
            url = doc.get("url") or doc.get("href") or ""
            content = (
                doc.get("raw_content") or doc.get("content") or doc.get("snippet") or ""
            )
            header = f'<Document href="{url}"/>' if url else "<Document>"
            title_block = f"Title: {title}\n" if title else ""
            formatted_documents.append(f"{header}\n{title_block}{content}\n</Document>")
            continue
        formatted_documents.append(f"<Document>\n{doc}\n</Document>")

    formatted_search_docs = "\n\n---\n\n".join(formatted_documents)

    return {"found_documents": [[formatted_search_docs]]}


def generate_report(state: ResearcherInstanceInternalState):
    previous_messages = state["messages"]
    found_documents = found_documents = [
        doc for iteration_docs in state["found_documents"] for doc in iteration_docs
    ]

    formatted_instructions = SystemMessage(
        content=generate_report_instructions.content.format(
            found_documents=found_documents,
            opportunity_type=state["opportunity_type"],
        )
    )
    result = llm.invoke([formatted_instructions] + previous_messages)
    # will override previous reports
    return {"researcher_reports": [result.content]}


def assess_results_quality(state: ResearcherInstanceInternalState):
    structured_llm = llm.with_structured_output(ResultsQualityAssesment)
    message = assess_quality_instruction.format(
        opportunity_type=state["opportunity_type"],
        additional_context=state["additional_context"],
        search_results_summarized=state["researcher_reports"],
    )
    result = structured_llm.invoke([SystemMessage(content=message)])
    return {
        "last_assessment_results": result,
        "iterations_count": state.get("iterations_count", 0) + 1,
    }


def finish_research_or_iterate(state: ResearcherInstanceInternalState):
    last_assessment_results = state["last_assessment_results"]
    if (
        last_assessment_results.is_sufficient
        or state["iterations_count"]
        >= 1  # turning off for now, as was getting too many docs to summarize, leading to context limits
    ):
        return "__end__"
    else:
        return "search_web"


researcher_builder = StateGraph(ResearcherInstanceInternalState)
researcher_builder.add_node("search_web", search_web)
researcher_builder.add_node("generate_report", generate_report)
researcher_builder.add_node("assess_results_quality", assess_results_quality)

researcher_builder.add_edge(START, "search_web")
researcher_builder.add_edge("search_web", "generate_report")
researcher_builder.add_edge("generate_report", "assess_results_quality")
researcher_builder.add_conditional_edges(
    "assess_results_quality",
    finish_research_or_iterate,
)

identifying_research_tasks_instructions = SystemMessage(
    """
    You're an assistant asked to help with searching different opportunities in Gen AI and Machine learning field.
    Your first task is to identify the types of opportunities user is searching for, and also some additional context,
    like within which time period, location, offline/online, etc.

    Until other types or other context is provided, use these as your defaults:

    Types:
    - conferences
    - hackathons
    xx intensives (exclude for now)
    - events
    xx summer / winter programs by big tech companies (exclude for now)

    Time frame:
    Within next 6 months

    Location:
    Belgrade, Serbia. For intensives, winter/summer programs - it can be online

    Also, here're previous tasks you've identified:
    {previous_tasks}
    And human feedback on these tasks:
    {human_feedback_on_tasks}

    Also, keep in mind current rules and preferences that user has when identifying tasks and providing context:
    {existing_rules_and_preferences}

    If empty, just ignore them.
    
    """
)

summarize_research_results_instructions = HumanMessage(
    """
    You'll be given list of search results for different types of opportunities user was searching for. Summarize search results for different types of opportunities into a structured report.
    Divide it into sections, highlight some most important events.
    Do not omit any of the opportunities returned by web researchers
    Here's the list of search results:
    {search_results}
    """
)

remembering_rules_and_prefences_instruction = """
You're an assistant asked to help with searching different opportunities in Gen AI and Machine learning field. 
Reflect on latest user feedback, if any changes to current rules and prefences are needed.
If user gives some general feedback, e.g. "do not capture data from these resources", or "give more events from x company",
capture it in rules and preferences. If no updates are needed, e.g. user just didn't like the output without specific general changes in guidelines,
just do not update anything. If no change needed, you can just flag it and keep updated rules empty (they will be kept as is now).

Current memory:
{current_memory}

Latest human feedback on search results:
{latest_feedback}

Results on which feedback was given:
{results_on_which_feedback_was_given}
"""


class ResearchTask(BaseModel):
    opportunity_type: str
    additional_context: str


class ResearchTasksList(BaseModel):
    tasks: list[ResearchTask]


class MlOpportunitiesResearcherState(MessagesState):
    research_tasks: list[ResearchTask]
    human_feedback_on_tasks: str
    researcher_reports: Annotated[list, operator.add]
    opportunities_report: str
    human_feedback_final_result: str


class MemoryUpdate(BaseModel):
    change_needed: bool
    updated_rules: str


def identify_research_tasks(
    state: MlOpportunitiesResearcherState, config: RunnableConfig, store: BaseStore
):
    user_id = config["configurable"]["user_id"]

    namespace = ("memory", user_id)
    key = "rules_and_preferences"

    existing_rules_and_preferences_item = store.get(namespace, key)
    existing_rules_and_preferences = (
        existing_rules_and_preferences_item.value.get("rules")
        if existing_rules_and_preferences_item
        else None
    )

    previous_messages = state["messages"]
    print(f"previous_messages: {previous_messages}")

    structured_llm = llm.with_structured_output(ResearchTasksList)
    message = identifying_research_tasks_instructions.content.format(
        previous_tasks=state.get("research_tasks"),
        human_feedback_on_tasks=state.get("human_feedback_on_tasks"),
        existing_rules_and_preferences=existing_rules_and_preferences,
    )
    result = structured_llm.invoke([SystemMessage(content=message)] + previous_messages)
    return {
        "research_tasks": result.tasks,
        # "messages": [result],
    }


def human_feedback(state: MlOpportunitiesResearcherState):
    pass


def initiate_researches_or_iterate_to_tasks(state: MlOpportunitiesResearcherState):
    human_feedback = state.get("human_feedback_on_tasks")
    if human_feedback is None or human_feedback.lower() == "approve":
        research_tasks = state["research_tasks"]
        return [
            Send(
                "conduct_research",
                arg={
                    "opportunity_type": r_task.opportunity_type,
                    "additional_context": r_task.additional_context,
                },
            )
            for r_task in research_tasks
        ]
    else:
        return "identify_research_tasks"


def summarize_final_result(state: MlOpportunitiesResearcherState):
    previous_messages = state["messages"]
    research_results = state["researcher_reports"]
    formatted_instructions = HumanMessage(
        content=summarize_research_results_instructions.content.format(
            search_results=research_results
        )
    )
    result = llm.invoke(previous_messages + [formatted_instructions])
    return {"opportunities_report": result.content}


def human_feedback_final_result(state: MlOpportunitiesResearcherState):
    pass


def route_to_end_or_rerun_research(state: MlOpportunitiesResearcherState):
    human_feedback_final_result = state.get("human_feedback_final_result")
    if (
        not human_feedback_final_result
        or human_feedback_final_result.lower() == "approve"
    ):
        # route to saving memory node
        return "continue"
    else:
        return "iterate"


def save_general_rules_and_preferences(
    state: MlOpportunitiesResearcherState, config: RunnableConfig, store: BaseStore
):
    user_id = config["configurable"]["user_id"]

    namespace = ("memory", user_id)
    key = "rules_and_preferences"

    # TODO try trustcall and collection storage
    existing_rules_and_preferences_item = store.get(namespace, key)
    existing_rules_and_preferences = (
        existing_rules_and_preferences_item.value.get("rules")
        if existing_rules_and_preferences_item
        else None
    )

    latest_human_feedback = state["human_feedback_final_result"]

    message = remembering_rules_and_prefences_instruction.format(
        current_memory=existing_rules_and_preferences,
        latest_feedback=latest_human_feedback,
        results_on_which_feedback_was_given=state["opportunities_report"],
    )

    structured_llm = llm.with_structured_output(MemoryUpdate)
    result: MemoryUpdate = structured_llm.invoke([SystemMessage(content=message)])
    if result.change_needed:
        store.put(namespace, key, {"rules": result.updated_rules})

    return None


ml_opportunities_researcher_builder = StateGraph(MlOpportunitiesResearcherState)
ml_opportunities_researcher_builder.add_node(
    "identify_research_tasks", identify_research_tasks
)
ml_opportunities_researcher_builder.add_node("human_feedback", human_feedback)
ml_opportunities_researcher_builder.add_node(
    "conduct_research", researcher_builder.compile()
)
ml_opportunities_researcher_builder.add_node(
    "summarize_final_result", summarize_final_result
)
ml_opportunities_researcher_builder.add_node(
    "human_feedback_final_result", human_feedback_final_result
)
ml_opportunities_researcher_builder.add_node(
    "save_general_rules_and_preferences", save_general_rules_and_preferences
)

ml_opportunities_researcher_builder.add_edge(START, "identify_research_tasks")
ml_opportunities_researcher_builder.add_edge(
    "identify_research_tasks", "human_feedback"
)
# https://docs.langchain.com/oss/python/langgraph/graph-api#conditional-edges
ml_opportunities_researcher_builder.add_conditional_edges(
    "human_feedback",
    initiate_researches_or_iterate_to_tasks,
    {
        "identify_research_tasks": "identify_research_tasks",
        "conduct_research": "conduct_research",
    },
)
ml_opportunities_researcher_builder.add_edge(
    "conduct_research", "summarize_final_result"
)
ml_opportunities_researcher_builder.add_edge(
    "summarize_final_result", "human_feedback_final_result"
)
ml_opportunities_researcher_builder.add_conditional_edges(
    "human_feedback_final_result",
    route_to_end_or_rerun_research,
    {
        "continue": END,
        "iterate": "save_general_rules_and_preferences",
    },
)
ml_opportunities_researcher_builder.add_edge(
    "save_general_rules_and_preferences", "identify_research_tasks"
)


# Memory savers
# ================================
sqlite_conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

postgres_conn_string = "postgresql://postgres:mysecret@localhost:5432/langchain-db"
postgres_conn = Connection.connect(postgres_conn_string)
postgres_conn.autocommit = True
postgres_store = PostgresStore(postgres_conn)

# ================================


graph = ml_opportunities_researcher_builder.compile(
    # interrupt_before=["human_feedback"], 
    interrupt_before=["human_feedback_final_result"],
    checkpointer=memory,
    store=postgres_store,
)  # fmt: skip


# config = {"configurable": {"thread_id": "15", "user_id": "Andrei researcher"}}
# graph.invoke(
#     {"messages": HumanMessage("Hey what are opportunities in nearest 3 months?")},
#     config=config,
# )
# graph.update_state(config, {"human_feedback_on_tasks": None}, as_node="human_feedback")
# graph.invoke(None, config)
# graph.get_state(config).next

# tavily_search = TavilySearch(max_results=3, include_raw_content=True)

# query = {
#     "query": "Please specify: 1) type of opportunity (hackathon, internship, fellowship, competition, conference/workshop, bootcamp, course), 2) timeframe (e.g., within next 3 months), 3) location (online or city/country), 4) your experience level (student/early-career/experienced), and 5) focus area if any (generative AI, NLP, CV, RL, etc.)."
# }
# res = tavily_search.invoke({"query": query["query"]})


# {"messages": [{"role": "user", "content": "Opportunities in next 3 months"}]}

# graph.invoke(
#     {"messages": HumanMessage("Hey what are hackathons in nearest 3 months?")},
#     config=config,
# )


# feedback = """Please focus on specific events type, sometimes for hackathon e.g. you return some meetup. Also, do not provide just general listing, as it mean i would need to search by myself though it. But ok to include it as separate resources to explore section. Also, please for hackathons and meetups check yandex also, as they have high quality events"""
# graph.update_state(config, {"human_feedback_final_result": feedback}, as_node = "human_feedback_final_result")
