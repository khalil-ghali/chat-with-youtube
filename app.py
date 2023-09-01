
import streamlit as st
hide_streamlit_style = """
            <style>
            #MainMenu  {visibility: hidden;}
            footer  {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
def paid_version(): 
    import os
    import argparse
    import shutil
    from langchain.document_loaders import YoutubeLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.chains import RetrievalQA
    from langchain.llms import OpenAI
    import streamlit as st
    from langchain.chat_models import ChatOpenAI
    from urllib.parse import urlparse, parse_qs
    def extract_video_id(youtube_url):
        try:
            parsed_url = urlparse(youtube_url)
            query_params = parse_qs(parsed_url.query)
            video_id = query_params.get('v', [None])[0]
            
            return video_id
        except Exception as e:
            print(f"Error extracting video ID: {e}")
            return None
    def set_openAi_api_key(api_key: str):
                st.session_state["OPENAI_API_KEY"] = api_key
                os.environ['OPENAI_API_KEY'] = api_key
    def openai_api_insert_component():
                with st.sidebar:
                    st.markdown(
                        """
                        ## Quick Guide 🚀
                        1. Get started by adding your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑
                        2. Easily input the video url
                        3. Engage with the content - ask questions, seek answers💬
                        """
                    )

                    api_key_input = st.text_input("Input your OpenAI API Key",
                                                type="password",
                                                placeholder="Format: sk-...",
                                                help="You can get your API key from https://platform.openai.com/account/api-keys.")
                    
                    
                    if api_key_input == "" or api_key_input is None:
                            st.sidebar.caption("👆 :red[Please set your OpenAI API Key here]")
                    
                    
                    st.caption(":green[Your API is not stored anywhere. It is only used to generate answers to your questions.]")

                    set_openAi_api_key(api_key_input)

    def launchpaidversion():
            openai_api_insert_component()
            os.environ['OPENAI_API_KEY'] = st.session_state['OPENAI_API_KEY']
            st.title('MKG: Your Chat with Website Assistant')
            st.subheader('Input your website URL, ask questions, and receive answers directly from the website.')

            videourl = st.text_input("Insert The video URL") 
            query = st.text_input("Ask any question about the video") 
            if st.button("Submit Question", type="primary"):
                video_id = extract_video_id(videourl)
                loader = YoutubeLoader(video_id)
                documents = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                documents = text_splitter.split_documents(documents)

                shutil.rmtree('./data')
                vectordb = Chroma.from_documents(
                    documents,
                    embedding=OpenAIEmbeddings(),
                    persist_directory='./data'
            )
                vectordb.persist()

                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatOpenAI(model_name='gpt-3.5-turbo'),
                    retriever=vectordb.as_retriever(),
                    return_source_documents=True,
                    verbose=False
                )
                response = qa_chain(query)
                st.write(response)

    
    launchpaidversion()
def free_version():
          import torch
          import os
          import argparse
          import shutil
          from langchain.document_loaders import YoutubeLoader
          from langchain.text_splitter import RecursiveCharacterTextSplitter
          from langchain.vectorstores import Chroma
          from langchain.embeddings import OpenAIEmbeddings
          from langchain.chains import RetrievalQA
          from langchain.llms import OpenAI
          import streamlit as st
          from langchain.chat_models import ChatOpenAI
          from langchain import HuggingFaceHub
          from langchain.embeddings import HuggingFaceInstructEmbeddings
          from urllib.parse import urlparse, parse_qs
          from langchain.embeddings import HuggingFaceBgeEmbeddings
          from transformers import pipeline
          import textwrap
          import time
          from deep_translator import GoogleTranslator
          from langdetect import detect


          def typewriter(text: str, speed: float):
                      container = st.empty()
                      displayed_text = ""

                      for char in text:
                          displayed_text += char
                          container.markdown(displayed_text)
                          time.sleep(1/speed)
          def wrap_text_preserve_newlines(text, width=110):
                      # Split the input text into lines based on newline characters
                      lines = text.split('\n')

                      # Wrap each line individually
                      wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

                      # Join the wrapped lines back together using newline characters
                      wrapped_text = '\n'.join(wrapped_lines)
                      return wrapped_text
          def process_llm_response(llm_originalresponse2):
                      #result_text = wrap_text_preserve_newlines(llm_originalresponse2["result"])
                      typewriter(llm_originalresponse2["result"], speed=40)

          def extract_video_id(youtube_url):
              try:
                  parsed_url = urlparse(youtube_url)
                  query_params = parse_qs(parsed_url.query)
                  video_id = query_params.get('v', [None])[0]

                  return video_id
              except Exception as e:
                  print(f"Error extracting video ID: {e}")
                  return None
          def set_openAi_api_key(api_key: str):
                      st.session_state["OPENAI_API_KEY"] = api_key
                      os.environ['OPENAI_API_KEY'] = api_key
          def openai_api_insert_component():
                      with st.sidebar:
                          st.markdown(
                              """
                              ## Quick Guide 🚀
                              1. Get started by adding your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑
                              2. Easily input the video url
                              3. Engage with the content - ask questions, seek answers💬
                              """
                          )

                          api_key_input = st.text_input("Input your OpenAI API Key",
                                                      type="password",
                                                      placeholder="Format: sk-...",
                                                      help="You can get your API key from https://platform.openai.com/account/api-keys.")


                          if api_key_input == "" or api_key_input is None:
                                  st.sidebar.caption("👆 :red[Please set your OpenAI API Key here]")


                          st.caption(":green[Your API is not stored anywhere. It is only used to generate answers to your questions.]")

                          set_openAi_api_key(api_key_input)

          def launchfreeversion():
                  HUGGINGFACE_API_TOKEN = "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"
                  model_name = "BAAI/bge-base-en"
                  encode_kwargs = {'normalize_embeddings': True}

                  st.title('MKG: Your Chat with Youtube Assistant')

                  videourl = st.text_input("Insert The video URL",  placeholder="Format should be like: https://www.youtube.com/watch?v=pSLeYvld8Mk")
                  query = st.text_input("Ask any question about the video",help="Suggested queries: Summarize the key points of this video - What is this video about - Ask about a specific thing in the video ")
                  st.warning("⚠️ Please Keep in mind that the accuracy of the response relies on the :red[Video's quality] and the :red[prompt's Quality]. Occasionally, the response may not be entirely accurate. Consider using the response as a reference rather than a definitive answer.")

                  if st.button("Submit Question", type="primary"):
                    with st.spinner('Processing the Video...'):
                        video_id = extract_video_id(videourl)
                        loader = YoutubeLoader(video_id)
                        documents = loader.load()

                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                        documents = text_splitter.split_documents(documents)
                        if os.path.exists('./data'):
                            shutil.rmtree('./data')
                        vectordb = Chroma.from_documents(
                        documents,
                        #embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                                                              # model_kwargs={"device": "cuda"})
                        embedding= HuggingFaceBgeEmbeddings( model_name=model_name, model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}, encode_kwargs=encode_kwargs)
                    )

                        repo_id = "tiiuae/falcon-7b-instruct"
                        qa_chain = RetrievalQA.from_chain_type(

                        llm=HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
                                        repo_id=repo_id,
                                        model_kwargs={"temperature":0.5, "max_new_tokens":1000}),
                            retriever=vectordb.as_retriever(),
                            return_source_documents=False,
                            verbose=False
                        )
                    with st.spinner('Generating Answer...'):
                          llm_response = qa_chain(query)
                          #llm_originalresponse2=llm_response['result']
                          process_llm_response(llm_response)
          launchfreeversion()


def intro():
            st.markdown("""
            # MKG: Your Chat with Youtube Assistant 🎬🤖

            Welcome to MKG-Assistant, where AI meets Youtube! 🚀🔍

            ## Base Models

            Q&A-Assistant is built on OpenAI's GPT 3.5 for the premium version and Falcon 7B instruct Model for the free version to enhance your websites browsing experience. Whether you're a student, researcher, or professional, we're here to simplify your interactions with the web. 💡📚

            ## How to Get Started

            1.Enter the Video URL.
            2. Enter your API key.(Only if you chose the premium version. Key is not needed in the free version)
            3. Ask questions using everyday language.
            4. Get detailed, AI-generated answers.

            5. Enjoy a smarter way to Interact with Youtube!



            ## It is Time to Dive in!


            """)                
page_names_to_funcs = {
    "Main Page": intro,
    "Open Source Edition (Free version)": free_version,
    "Premium edition (Requires Open AI API Key )": paid_version
    
}


    
    
    
    

demo_name = st.sidebar.selectbox("Choose a version", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()    
st.sidebar.markdown('<a href="https://www.linkedin.com/in/mohammed-khalil-ghali-11305119b/"> Connect on LinkedIn <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" alt="LinkedIn" width="30" height="30"></a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="https://github.com/khalil-ghali"> Check out my GitHub <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" alt="GitHub" width="30" height="30"></a>', unsafe_allow_html=True)
