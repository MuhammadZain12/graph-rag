from setuptools import setup, find_packages

setup(
    name="graph_rag_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "neo4j",
        "vllm"
    ],
)
