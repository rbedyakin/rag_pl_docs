# -*- coding: utf-8 -*-

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from typing import Any, List, Optional
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.typesense import TypesenseVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
from typesense import Client as TypesenseClient
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader


class VectorDBRetriever(BaseRetriever):
    """Retriever over a typesense vector store."""

    def __init__(
        self,
        vector_store: TypesenseVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = self._embed_model.get_query_embedding(
            query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores


class RAG:

    def __init__(self,
                 embed_model_name: str = "BAAI/bge-small-en-v1.5",
                 llm_model: str = 'llama3.2:3b'):

        self.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

        self.llm = Ollama(
            model=llm_model,
            temperature=0.1,
            request_timeout=60.0,
            # max_new_tokens=256,
            # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
            # context_window=3900,
            # kwargs to pass to __call__()
            # generate_kwargs={},
            # kwargs to pass to __init__()
            # set to at least 1 to use GPU
            # model_kwargs={"n_gpu_layers": 1},
            # verbose=True,
        )

        self.typesense_client = TypesenseClient({
            "api_key":
            "xyz",
            "nodes": [{
                "host": "localhost",
                "port": "8108",
                "protocol": "http"
            }],
            "connection_timeout_seconds":
            2,
        })

        self.vector_store = TypesenseVectorStore(
            client=self.typesense_client,
            collection_name='lightningdocs',
            text_key='textchunks')

        self.text_parser = SentenceSplitter(
            chunk_size=1024,
            separator=" ",
        )

        self.retriever = VectorDBRetriever(vector_store=self.vector_store,
                                           embed_model=self.embed_model,
                                           query_mode="default",
                                           similarity_top_k=2)
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.retriever, llm=self.llm)

    def load_data(self, docs_dir: str):
        reader = SimpleDirectoryReader(input_dir=docs_dir,
                                       required_exts=[
                                           '.rst',
                                       ],
                                       recursive=True)
        documents = reader.load_data()

        self.add_documents(documents)

    def add_documents(self, documents):
        text_chunks = []
        doc_idxs = []
        for doc_idx, doc in enumerate(documents):
            cur_text_chunks = self.text_parser.split_text(doc.text)
            text_chunks.extend(cur_text_chunks)
            doc_idxs.extend([doc_idx] * len(cur_text_chunks))

        nodes = []
        for idx, text_chunk in enumerate(text_chunks):
            node = TextNode(text=text_chunk, )
            src_doc = documents[doc_idxs[idx]]
            node.metadata = src_doc.metadata
            nodes.append(node)

        for node in nodes:
            node_embedding = self.embed_model.get_text_embedding(
                node.get_content(metadata_mode="all"))
            node.embedding = node_embedding

        self.vector_store.add(nodes)

    def query(self, query_str: str):
        response = self.query_engine.query(query_str)
        return response


if __name__ == "__main__":
    rag = RAG(embed_model_name="BAAI/bge-small-en-v1.5",
              llm_model='llama3.2:3b')
    rag.load_data(docs_dir="./data/pytorch-lightning-release-stable/docs/")

    query_strs = [
        "What is PyTorch Lightning?",
        "How to train the model in Pytorch Lightning?",
        "How to deploy a model?"
    ]
    for query_str in query_strs:
        print(f'Question: {query_str}')
        response = rag.query(query_str=query_str)

        print(f'Answer: {str(response)}')
        print()
        # print(response.source_nodes[0].get_content())
