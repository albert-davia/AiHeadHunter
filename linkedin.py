from operator import add
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from davia import Davia


load_dotenv()


class PersonInfo(TypedDict):
    name: str
    location: str
    job: str
    company: str
    justifications: str
    linkedin: str


class HRRecruiter(MessagesState):
    info: Annotated[list[PersonInfo], add]


model = ChatOpenAI(
    model="gpt-4o",
)


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls: # type: ignore
        return "tools"
    return END


@tool
def get_person_info(
    name_person: str,
    topic: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Call to get information about a person."""
    internet_search = TavilySearch(
        max_results=10,
    )

    info_about_person = internet_search.invoke(
        {"query": f"{name_person}, {topic} , location, job"},
        stream=False,
    )

    response = model.with_structured_output(PersonInfo).invoke(
        [
            SystemMessage(
                content="""You are a helpful assistant tasked with gathering information about a person.
Your objective is to retrieve the following details:
	•	Name
	•	Location
	•	Job
    •	Company
    •	Justifications: List key job-relevant strengths in the shortest possible bullet points. Keep it extremely concise.
    •	Link to the LinkedIn profile
                """
            ),
            HumanMessage(
                content=f"Here is the information I found on the internet about {name_person}: {info_about_person}"
            ),
        ],
        stream=False,
    )

    return Command(
        update={
            # update the state keys
            "info": [response],
            # update the message history
            "messages": [
                ToolMessage(
                    f"Successfully looked up user information for {name_person}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )
    

internet_search = TavilySearch(
    max_results=15,
    include_domains=["https://www.linkedin.com/"],
)

tool_node = ToolNode(tools=[get_person_info, internet_search])


def chat(state: HRRecruiter):
    model_with_tools = model.bind_tools([get_person_info, internet_search])
    current_info = state["info"]
    messages = state["messages"]

    history_length = 8  # Minimum number of messages to keep
    last_messages = messages[-history_length:]
    non_tool_indices = [
        i for i, m in enumerate(messages) if not isinstance(m, ToolMessage)
    ]

    # If first message is a tool message, extend back to last non-tool message
    if isinstance(last_messages[0], ToolMessage):
        start_idx = next(
            i for i in reversed(non_tool_indices) if i <= len(messages) - history_length
        )
        last_messages = messages[start_idx:]

    response = model_with_tools.invoke(
        [
            SystemMessage(
                content=f"""You are an efficient recruiter assistant.
                        Your task is to build a list of people to interview. To achieve this, you will:
                        • Use the internet_search tool multiple times with different queries, refining them iteratively to find the most relevant names. Do not use the same query twice.
                        • Identify multiple relevant individuals based on the user's criteria. The final goal is people, not companies. While you can perform broader searches if needed, ensure that the final result consists of individual people. 
                        • The internet searches can be detailed if necessary.
                        • For each identified person, call get_person_info separately to update the interview list, using their name and the relevant topic.
                        • For each user query, do not retrieve more than 10 people.

                        Current table state:
                        {current_info}

                        Keep responses strictly concise. ONLY output what you have just done—no explanations, summaries, or extra details. DO NOT EXPLAIN ANYTHING.
                                        """
            ),
            *last_messages,
        ],
    )

    return {
        "messages": [response],
    }


app = Davia(title="HR Recruiter")

@app.graph
def graph():
    graph = StateGraph(HRRecruiter)
    graph.add_node("chat", chat)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", should_continue, ["tools", END])
    graph.add_edge("tools", "chat")
    return graph


if __name__ == "__main__":
    app.run()
