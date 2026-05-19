from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("../research paper.pdf")
documents = loader.load()

