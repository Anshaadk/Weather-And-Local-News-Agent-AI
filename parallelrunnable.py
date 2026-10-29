from dotenv import load_dotenv
load_dotenv()

# from langchain_mistralai import ChatMistralAI
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda





model = ChatGroq(model="openai/gpt-oss-120b")
parser = StrOutputParser()

short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 line"
)

detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)


# topic = "Mechine Learning"

chain = RunnableParallel({
    "short": RunnableLambda(lambda x:x["short"]) | short_prompt | model | parser,
    "detailed": RunnableLambda(lambda x:x["detailed"]) | detailed_prompt | model | parser
})

result =chain.invoke({
    "short" : {
        "topic": "Mechine Learning"
    },
    "detailed" : {
        "topic": "Deep Learning"
    }
})

print(result) 