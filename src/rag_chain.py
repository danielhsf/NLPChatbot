import os
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import LLM_MODEL

load_dotenv()

SYSTEM_PROMPT = """You are an expert NLP assistant. Answer questions using ONLY the context provided below.
For each piece of information you use, cite the source and page number in the format: [Source: filename, Page: X].
If the answer is not found in the context, say "I don't know based on the provided references."

Context:
{context}"""


def format_docs(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        filename = os.path.basename(source)
        parts.append(f"[{i}] (Source: {filename}, Page: {page})\n{doc.page_content}")
    return "\n\n".join(parts)


def build_rag_chain(retriever):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    llm = ChatAnthropic(model=LLM_MODEL)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever
