from models.llm import LLM
from models.pt_llm import PromptTuningLLM
from models.graph_llm import GraphLLM


load_model = {
    #"llm": LLM,
    # "inference_llm": LLM,
    # "pt_llm": PromptTuningLLM,
    "graph_llm": GraphLLM,
}

# Replace the following with the model paths
llama_model_path = {
    "7b": "edumunozsala/llama-2-7b-int4-python-code-20k",
    #"7b": "meta-llama/Llama-2-7b-hf",
    # "7b_chat": "meta-llama/Llama-2-7b-chat-hf",
    # "13b": "meta-llama/Llama-2-13b-hf",
    # "13b_chat": "meta-llama/Llama-2-13b-chat-hf",
}