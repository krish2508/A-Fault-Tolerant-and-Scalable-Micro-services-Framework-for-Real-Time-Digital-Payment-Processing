from vector import vector_store
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
logging.basicConfig(

    filename='example.log', format='%(asctime)s - %(levelname)s - %(message)s',
                    )
retriever = vector_store.as_retriever()
memory = InMemoryChatMessageHistory()  

while True:
    breakpoint()
    query = input("Enter your question: ")
    # breakpoint()
    if query.lower() in ['exit', 'quit']:
        print("Exiting the program.")
        break
    results = retriever.invoke(query)
    # print(type(results))
    context = "\n".join([doc.page_content for doc in results])
    history = "\n".join([f"{m.type}: {m.content}" for m in memory.messages])
    prompt = f"""
    You are a helpful assistant.

    Answer the question based ONLY on the context below.

    Context:
    {context}
    Conversation History:
    {history}

    Question:
    {query}
    """
    # breakpoint()
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-3.1-flash-lite",
        temperature=0
    )

    response = llm.invoke(prompt)
    print(response.content)
    memory.add_user_message(query)
    memory.add_ai_message(response.content)