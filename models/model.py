import lmstudio as lms

class LLModel:
    
    def __init__(self, name: str) -> None:
        self.llm = lms.llm(name)
        self.name = name
        self.history = []
        self.chat = lms.Chat()


    def respond(self, prompt) -> lms.PredictionResult:
        response: lms.PredictionResult = self.llm.respond(prompt)
        return response

    def respond_stream(self, prompt: str, raw: bool = False) -> None:
        try:
            self.chat.add_user_message(prompt)
            response_stream: lms.PredictionStream = self.llm.respond_stream(self.chat)
            for partial_response in response_stream:
                if raw:
                    print(partial_response, flush=True)
                else:
                    print(partial_response.content, end="", flush=True)  # Muestra cada parte del mensaje en tiempo real
        except Exception as e:
            print(f"Error al responder: {e}")
            raise e

    def set_system_context(self, context_prompt: str) -> None:
        self.chat = lms.Chat(context_prompt)
