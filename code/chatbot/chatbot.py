import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from llama_cpp import Llama

class EmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        pass

class SentenceTransformerEmbedding(EmbeddingModel):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts)

class HuggingFaceEmbedding(EmbeddingModel):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].tolist()

class ContextSearchEngine:
    def __init__(self, embedding_model: EmbeddingModel, chunks: List[str]):
        self.embedding_model = embedding_model
        self.chunks = chunks
        self.context_embeddings = self.generate_embeddings(chunks)

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        return self.embedding_model.encode(chunks)

    def find_context(self, question: str, top_k: int = 3) -> str:
        query_embedding = self.embedding_model.encode([f"Represent this sentence for searching relevant passages: {question}"])[0]
        scores = util.dot_score(torch.tensor([query_embedding]), torch.tensor(self.context_embeddings))[0].tolist()
        doc_score_pairs = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)
        return "\n".join([doc for doc, _ in doc_score_pairs[:top_k]])

class LanguageModel(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, stop: List[str]) -> str:
        pass

class LlamaCppModel(LanguageModel):
    def __init__(self, model_path: str, n_ctx: int = 2000, n_threads: int = 8, n_gpu_layers: int = 0):
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            n_batch=2
        )

    def generate(self, prompt: str, max_tokens: int, stop: List[str]) -> str:
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            stop=stop,
            echo=False
        )
        return output['choices'][0]['text'].strip()

class ChatbotPromptManager:
    def __init__(self):
        self.prompts = {
            "GENERAL": """<|system|>
You are a helpful AI Assistant developed by Trellissoft in the year 2024. You are given context information below.
======
{context}
======
### Instructions Start
1. Whenever the user question includes 'you' or 'your', consider it as first person on behalf of `Trellissoft`.
2. You will phrase the answer in first person using words such as We, us, etc.
3. Keep your tone positive, clear, very brief, and very short.
4. Do not add unnecessary information. Keep your response very brief and concise.
### Instructions End
Your task is to have an engaging conversation with the user using the above context information and instructions, continue the below chat conversation with the user.
Keep your response very brief, short, and to the point.<|end|>
{chat_history}
<|user|>
{question}<|end|>
<|assistant|>"""
        }

    def get_prompt(self, key: str, **kwargs) -> str:
        return self.prompts[key].format(**kwargs)

class Chatbot:
    def __init__(self, context_search: ContextSearchEngine, language_model: LanguageModel, prompt_manager: ChatbotPromptManager):
        self.context_search = context_search
        self.language_model = language_model
        self.prompt_manager = prompt_manager
        self.chat_history = []

    def generate_response(self, question: str) -> str:
        context = self.context_search.find_context(question)
        prompt = self.prompt_manager.get_prompt(
            "GENERAL",
            context=context,
            chat_history="\n".join(self.chat_history),
            question=question
        )
        response = self.language_model.generate(prompt, max_tokens=150, stop=["<|end|>"])
        self.chat_history.append(f"<|user|>\n{question}<|end|>")
        self.chat_history.append(f"<|assistant|>\n{response}<|end|>")
        return response

def main():
    # Initialize components
    embedding_model = SentenceTransformerEmbedding(os.path.join(os.environ['MOUNTED_MODEL_DIRECTORY'], "embedding-model"))
    context_chunks = ["Sample context 1", "Sample context 2", "Sample context 3"]  # Replace with actual context
    context_search = ContextSearchEngine(embedding_model, context_chunks)

    model_path = os.path.join(os.environ['MOUNTED_MODEL_DIRECTORY'], [m for m in os.listdir(os.environ['MOUNTED_MODEL_DIRECTORY']) if m.endswith(".gguf")][0])
    language_model = LlamaCppModel(model_path)

    prompt_manager = ChatbotPromptManager()

    # Create chatbot
    chatbot = Chatbot(context_search, language_model, prompt_manager)

    # Example usage
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
        response = chatbot.generate_response(user_input)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()