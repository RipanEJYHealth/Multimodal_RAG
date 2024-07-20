import argparse
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext
import qdrant_client

def create_multimodal_index(data_path, qdrant_path, persist_dir):
    # Create a local Qdrant vector store
    client = qdrant_client.QdrantClient(path=qdrant_path)

    text_store = QdrantVectorStore(
        client=client, collection_name="text_collection"
    )
    image_store = QdrantVectorStore(
        client=client, collection_name="image_collection"
    )
    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )

    # Create the MultiModal index
    documents = SimpleDirectoryReader(data_path).load_data()
    index = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    # Save it
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"Index created and saved at {persist_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a multimodal index using Qdrant and save it.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--qdrant_path", type=str, required=True, help="Path to the local Qdrant vector store.")
    parser.add_argument("--persist_dir", type=str, required=True, help="Directory to save the persisted index.")

    args = parser.parse_args()
    
    create_multimodal_index(args.data_path, args.qdrant_path, args.persist_dir)
