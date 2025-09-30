from smolagents import LiteLLMModel, CodeAgent, WebSearchTool
import ollama

# MODEL = 'ollama_chat/qwen2.5-coder:14b'
# MODEL = 'command-r7b:7b'
MODEL = "qwen3-coder:30b"

ollama.pull(MODEL)

model = LiteLLMModel(model_id=f"ollama_chat/{MODEL}")

agent = CodeAgent(tools=[WebSearchTool()], model=model, stream_outputs=True)

agent.run("What would it cost to administer a COVID-19 vaccine to the entire US population?")
