from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

if __name__ == "__main__":
    print("Hello LangChain!!")

    linkedin_profile_url = linkedin_lookup_agent(name="Shubham Jindal Deloitte")

    '''summary_template = """
      given the information {information} about a person from I want you to answer below:
      1. Is Asokha Hindu or A Buddist
      2. Did Ashoka marry queen of Kalinga kingdom
    """ '''

    summary_template = """
        given the LinkedIn information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_data = scrape_linkedin_profile(
        # linkedin_profile_url="https://www.linkedin.com/in/eden-marco/"
        linkedin_profile_url="https://gist.githubusercontent.com/pravyn25c/dca19386963f1dfa6345930981b5c7e8/raw/ab4159ce0ff4d25adfa2f37658d2b986eacac3bc/gistfile1.json"
    )

    print(chain.run(information=linkedin_data))

    print(linkedin_data.json())
