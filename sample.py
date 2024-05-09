import time

from crewai                               import Agent, Task, Crew, Process
from langchain_groq                       import ChatGroq
from langchain.vectorstores               import Chroma
from langchain_community.embeddings       import HuggingFaceEmbeddings

client_query   = "Can admin re-assign or un-assign any vendor from any group from the backend? If so then how ?"
agent_response = "No, the admin can't re-assign or un-assign any vendor from any group from the backend."

t1 = time.time()

response_verifier = Agent(
role='Senior QA Response Verifier',
goal='You will be given a client query and agent response along with a knowledgebase. You have to check whether the agent response is correct or not with respect to the client query and the context of knowledgebase',
backstory="""You work at a leading tech company Webkul.
Your expertise lies in checking whether the agent response made to a client query is actually correct or not with respect to the knowledgebase provided.""",
verbose=True,
allow_delegation=False,
llm=ChatGroq(model="llama3-70b-8192",temperature=0),
)

output_analyst = Agent(
role='Senior Output Analyst',
goal='take response from the response verifier agent check whether the response verifier agent claimed the response correct or not.',
backstory="""You work at a leading tech company Webkul.
Your expertise lies in checking whether the response verifier agent response claims the agent response given to it, correct or incorrect.""",
verbose=True,
allow_delegation=False,
llm=ChatGroq(model="llama3-70b-8192",temperature=0),
)

email_writer = Agent(
role='Email Writer Agent',
goal="""take response from the response verifier agent and 
write a helpful email to the Reporting Manager in a thoughtful and friendly way only if the response verifier agent claims 
that the response he gets is incorrect. If the response he gets is correct then don't write any email.

If the response is incorrect then simply mention the reason why the response is not correct as the response verifier agent will 
tell you the reason.

You never make up information. that hasn't been provided by the response verifier.
Always sign off the emails in appropriate manner and from Subhu Rawat the Machine Learning Department.
""",
backstory="""You are a master at synthesizing a variety of information got from response verifier and writing a helpful email
that will address the agent's mistake and provide them with helpful information as per response verifier output that is given to you.""",
llm=ChatGroq(model="llama3-70b-8192",temperature=0),
verbose=True,
allow_delegation=False,
# max_iter=5,
# memory=True,
)

response_verification = Task(
description="Given a client query {client_query} and the agent response {agent_response}. You have to check whether the agent response is correct or not against the client query with the context of knowledgebase {knowledgebase}. Correct means the agent response should match exactly with the knowledgebase data. Claim the agent response incorrect if the agent response is partial correct only.",
expected_output="correct or incorrect.",
agent=response_verifier
)

output_analysis = Task(
description="""Conduct a comprehensive analysis of the information provided from the response verifier and check if it claims agent response to be correct or not.
If the response verifier agent response is in correct context then simply give output as correct and if the response verifier agent
response is in incorrect context then simply give output as incorrect""",
expected_output="""correct! or incorrect! """,
context = [response_verification],
agent=output_analyst, 
)

email_drafting = Task(
description="""Conduct a comprehensive analysis of the information provided from the response verifier to write an email.
Write a simple, polite and to the point email which will explain the mistakes in agent response to the Reporting Manager.
If useful use the information provided from the response verifier.
Draft an email only if the response verifier claims that agent response is incorrect otherwise leave it blank.
""",
expected_output="""A well crafted email for the Reporting Manager that addresses mistakes in agent response as per response verifier output. But don't create any output file if the response verifier says that the agent response is correct.""",
context = [response_verification],
agent=email_writer,
output_file=f"draft_email.txt",
)

crew = Crew(
agents=[response_verifier,output_analyst],
tasks=[response_verification,output_analysis],
process=Process.sequential,
verbose=1,
full_output=True
)

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
db = Chroma(persist_directory='verification-knowledgebase/vectorstore',embedding_function=embeddings)
relevant_docs = db.similarity_search(client_query,k=10)
relevant_knowledgebase = []
for doc in relevant_docs:
    d = doc.page_content
    relevant_knowledgebase.append(d)

inputs = {'client_query':client_query ,'agent_response':agent_response,'knowledgebase':relevant_knowledgebase}
result = crew.kickoff(inputs=inputs)

if output_analysis.output.raw_output == 'incorrect!':
    email_drafting.execute()

print('-------------------------------------------------')
print('Client Query: ' + client_query)
print('-------------------------------------------------')
print('Agent Response: ' + agent_response)
print('-------------------------------------------------')
print(response_verification.output.raw_output)
print('-------------------------------------------------')
print(output_analysis.output.raw_output)
print('-------------------------------------------------')

t2 = time.time()
t = t2 - t1
print(round(t,2))
