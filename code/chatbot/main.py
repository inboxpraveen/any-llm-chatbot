import os
from chatbot import (
    SentenceTransformerEmbedding,
    ContextSearchEngine,
    LlamaCppModel,
    ChatbotPromptManager,
    Chatbot
)
from chatbot_grpc import serve_grpc
from chatbot_fastapi import serve_fastapi
import argparse
from multiprocessing import Process

def initialize_chatbot():
    embedding_model = SentenceTransformerEmbedding(os.path.join(os.environ['MOUNTED_MODEL_DIRECTORY'], "embedding-model"))
    context_chunks = ["Sample context 1", "Sample context 2", "Sample context 3"]  # Replace with actual context
    context_search = ContextSearchEngine(embedding_model, context_chunks)

    model_path = os.path.join(os.environ['MOUNTED_MODEL_DIRECTORY'], [m for m in os.listdir(os.environ['MOUNTED_MODEL_DIRECTORY']) if m.endswith(".gguf")][0])
    language_model = LlamaCppModel(model_path)

    prompt_manager = ChatbotPromptManager()

    return Chatbot(context_search, language_model, prompt_manager)

def main():
    parser = argparse.ArgumentParser(description="Run the chatbot server")
    parser.add_argument("--grpc_port", type=int, default=50051, help="gRPC server port")
    parser.add_argument("--fastapi_port", type=int, default=8000, help="FastAPI server port")
    parser.add_argument("--fastapi_host", type=str, default="0.0.0.0", help="FastAPI server host")
    args = parser.parse_args()

    chatbot = initialize_chatbot()

    grpc_process = Process(target=serve_grpc, args=(chatbot, args.grpc_port))
    fastapi_process = Process(target=serve_fastapi, args=(chatbot, args.fastapi_host, args.fastapi_port))

    grpc_process.start()
    fastapi_process.start()

    grpc_process.join()
    fastapi_process.join()

if __name__ == "__main__":
    main()
