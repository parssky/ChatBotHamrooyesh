from langchain_community.document_loaders import PyPDFium2Loader

loader = PyPDFium2Loader("computer_science_is_foundational.pdf")
data = loader.load()

print(data)
