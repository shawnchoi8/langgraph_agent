# %%
from typing_extensions import TypedDict
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    query: str  # user question
    answer: str  # ai answer (세율)
    tax_base_equation: str  # 과세표준 계산 수식
    tax_deduction: str  # 공제액
    market_ratio: str  # 공정시장가액비율
    tax_base: str  # 과세표준 계산


graph_builder = StateGraph(AgentState)

# %%
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

# embedding 되어있는 db 불러오기
vector_store = Chroma(
    embedding_function=embedding_function,
    collection_name="real_estate_tax",
    persist_directory="./real_estate_tax_collection",
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# %%
# 질문에 따라 dynamic하게 retrieval 가능
query = "5억짜리 집 1채, 10억짜리 집 1채, 20억짜리 집 1채를 가지고 있을 때, 세금을 얼마나 내나요?"

# %%
from langchain_openai import ChatOpenAI
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

client = Client()
rag_prompt = client.pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)

llm = ChatOpenAI(model="gpt-4o")
# 비용 절감을 위해 작은 llm model 선언
small_llm = ChatOpenAI(model="gpt-4o-mini")

# %%
# 과세표준 구하는 과정

# 여기서 나온 답변을 tax_base_equation_chain에 input으로 넣어
# 여기서 나온 결과 : 과세표준 공식을 글로 써준다.
# 이 결과가 tax_base_equation_chain의 tax_base_equation_information 로 들어간다
tax_base_retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

tax_base_equation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "사용자의 질문에서 과세표준을 계산하는 방법을 수식으로 나타내주세요. 부연설명 없이 수식만 나타내주세요.",
        ),
        ("human", "{tax_base_equation_information}"),
    ]
)

# tax_base_retrieval_chain의 결과가 들어온다.
tax_base_equation_chain = (
    {"tax_base_equation_information": RunnablePassthrough()}
    | tax_base_equation_prompt
    | llm
    | StrOutputParser()
)

# final chain
tax_base_chain = {
    "tax_base_equation_information": tax_base_retrieval_chain
} | tax_base_equation_chain


# 1. 과세표준 구하는 공식 그 자체를 구하는 노드
def get_tax_base_equation(state: AgentState):
    tax_base_equation_question = "주택에 대한 종합부동산세 계산시 과세표준을 계산하는 방법을 수식으로 표현해서 알려주세요"
    tax_base_equation = tax_base_chain.invoke(tax_base_equation_question)
    return {"tax_base_equation": tax_base_equation}


# %%
# 공제액 구하는 과정

# 공제액 구하는 chain
tax_deduction_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)


# 2. 과세표준의 공제 금액 구하는 노드 (retrieve)
def get_tax_deduction(state: AgentState):
    tax_deduction_question = "주택에 대한 종합 부동산세 계산시 공제금액을 알려주세요"
    tax_deduction = tax_deduction_chain.invoke(tax_deduction_question)
    return {"tax_deduction": tax_deduction}


# %%
# 공정시장가액비율 구하는 과정 -> 대통령령 -> 즉 web search가 필요하다.

from langchain_tavily import TavilySearch
from datetime import date

tavily_search_tool = TavilySearch(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

# system에 날짜 정보를 주지 않고, 아래 query에 날짜 정보를 넣는 것이 더 좋은 선택.
# 검색 자체에 날짜가 포함되는게 더 정확도가 높기 때문이다.
tax_market_ratio_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"아래 정보를 기반으로 공정시장 가액비율을 계산해주세요\n\nContext:\n{{context}}",
        ),
        ("human", "{query}"),
    ]
)


# 3. 공정시장가액비율 구하는 노드
def get_market_ratio(state: AgentState):
    query = f"오늘 날짜:({date.today()})에 해당하는주택 공시가격 공정시장가액비율은 몇%인가요?"
    context = tavily_search_tool.invoke(query)
    print(f"context == {context}")
    # define chain
    tax_market_ratio_chain = tax_market_ratio_prompt | llm | StrOutputParser()
    # 결과(chain에 invoke)
    market_ratio = tax_market_ratio_chain.invoke({"context": context, "query": query})
    return {"market_ratio": market_ratio}


# %%

tax_base_calculation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """주어진 내용을 기반으로 과세표준을 계산해주세요.
         과세표준 계산 공식: {tax_base_equation}
         공제금액: {tax_deduction}
         공정시장가액비율: {market_ratio}
         사용자 주택 공시가격 정보: {query}
         """,
        ),
        ("human", "사용자 주택 공시가격 정보: {query}"),
    ]
)


# 1,2,3 노드를 합쳐서 과세표준 계산
def calculate_tax_base(state: AgentState):
    tax_base_equation = state["tax_base_equation"]  # 공식
    tax_deduction = state["tax_deduction"]  # 공제 금액
    market_ratio = state["market_ratio"]  # 공정시장가액비율
    query = state["query"]
    # define chain
    tax_base_calculation_chain = tax_base_calculation_prompt | llm | StrOutputParser()
    # 결과 (과세표준 계산 결과)
    tax_base = tax_base_calculation_chain.invoke(
        {
            "tax_base_equation": tax_base_equation,
            "tax_deduction": tax_deduction,
            "market_ratio": market_ratio,
            "query": query,
        }
    )
    print(f"tax_base == {tax_base}")
    return {"tax_base": tax_base}


# %%
# 세율 계산
# 사용자의 query & 과세표준을 가지고 세율을 구한다음에
# 최종 계산을 해서 실제 세금으로 내야하는 금액 산출

tax_rate_calculation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 종합부동산세 계산 전문가 입니다. 아래 문서를 참고해서 사용자의 질문에 대한 종합부동산세를 계산해주세요
     종합부동산세 세율:{context}
     """,
        ),
        (
            "human",
            """과세표준과 사용자가 소지한 주택의 수가 아래와 같을 때 종합부동산세를 계산해주세요
     과세표준: {tax_base}
     주택수: {query}""",
        ),
    ]
)


# 세율 계산
def calculate_tax_rate(state: AgentState):
    query = state["query"]
    tax_base = state["tax_base"]
    context = retriever.invoke(query)
    # define chain
    tax_rate_chain = tax_rate_calculation_prompt | llm | StrOutputParser()
    # 결과 반환(invoke)
    tax_rate = tax_rate_chain.invoke(
        {"context": context, "tax_base": tax_base, "query": query}
    )
    print(f"tax_rate == {tax_rate}")
    return {"answer": tax_rate}


# %%
# node
graph_builder.add_node("get_tax_base_equation", get_tax_base_equation)
graph_builder.add_node("get_tax_deduction", get_tax_deduction)
graph_builder.add_node("get_market_ratio", get_market_ratio)
graph_builder.add_node("calculate_tax_base", calculate_tax_base)
graph_builder.add_node("calculate_tax_rate", calculate_tax_rate)

# %%
# edge
from langgraph.graph import START, END

graph_builder.add_edge(START, "get_tax_base_equation")
graph_builder.add_edge(START, "get_tax_deduction")
graph_builder.add_edge(START, "get_market_ratio")
graph_builder.add_edge("get_tax_base_equation", "calculate_tax_base")
graph_builder.add_edge("get_tax_deduction", "calculate_tax_base")
graph_builder.add_edge("get_market_ratio", "calculate_tax_base")
graph_builder.add_edge("calculate_tax_base", "calculate_tax_rate")
graph_builder.add_edge("calculate_tax_rate", END)

# %%
graph = graph_builder.compile()
