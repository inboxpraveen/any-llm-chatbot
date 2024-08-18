import grpc
from concurrent import futures
import chatbot_pb2
import chatbot_pb2_grpc
from chatbot import Chatbot

class ChatbotService(chatbot_pb2_grpc.ChatbotServicer):
    def __init__(self, chatbot: Chatbot):
        self.chatbot = chatbot

    def GenerateResponse(self, request, context):
        response = self.chatbot.generate_response(request.query)
        return chatbot_pb2.ChatResponse(response=response)

    def HealthCheck(self, request, context):
        return chatbot_pb2.HealthCheckResponse(status="OK")

def serve_grpc(chatbot: Chatbot, port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chatbot_pb2_grpc.add_ChatbotServicer_to_server(ChatbotService(chatbot), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"gRPC server started on port {port}")
    server.wait_for_termination()
