import pandas as pd
import bm25s

from bm25s.hf import BM25HF

from llmlingua import PromptCompressor
import pandas as pd
from tqdm import tqdm
#tqdm.pandas()

question='Please summarize the following discharge summary. You should not omit important information.'
#llm_lingua = PromptCompressor()
#llm_lingua = PromptCompressor(
#    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
#    use_llmlingua2=True, # Whether to use llmlingua-2
#)
def compress_text_method(prompt):

    
    #llm_lingua.model = llm_lingua.model.cuda()    
    compressed_prompt = llm_lingua.compress_prompt(
        [prompt],
        question=question,
        rate=0.3,
        # Set the special parameter for LongLLMLingua
        condition_in_question="after_condition",
        reorder_context="sort",
        dynamic_context_compression_ratio=0.3, # or 0.4
        condition_compare=True,
        context_budget="+100",
        rank_method="longllmlingua",
    )
    #llm_lingua.model = llm_lingua.model.cpu()
    #torch.cuda.empty_cache()
    print(compressed_prompt['compressed_tokens'])
    return compressed_prompt['compressed_prompt']



PROMPT_DICT = {
	"openended": {
		"gpt": (
			"Discharge Summary :\n"
			"{note}\n\n"
            #"Similar Patient Discharge Summary Context :\n"
            #"{context} \n\n"
			"Question : {question}\n\n"
			"Answer :"
        ),
		"llama-2-chat": (
			"[INST] <<SYS>>\n"
			"You are a helpful, respectful and honest assistant.\n"
			"<</SYS>>\n\n"
			"Discharge Summary :\n"
			"{note}\n\n"
            #"Similar Patient Discharge Summary Context :\n"
            #"{context} \n\n"
			"Question : {question}\n\n"
			"Answer : "
			"[/INST]"
        ),
	},
	"multichoice": {
		"gpt": (
			"Discharge Summary :\n"
			"{note}\n\n"
            #"Similar Patient Discharge Summary Context :\n"
            #"{context} \n\n"
			"Question : {question}\n"
			"Choices :\n"
			"A. {choice_a}\n"
			"B. {choice_b}\n"
			"C. {choice_c}\n"
			"D. {choice_d}\n"
			"E. {choice_e}\n\n"
			"Answer :"
		),
		"llama-2-chat": (
			"[INST] <<SYS>>\n"
			"You are a helpful, respectful and honest assistant.\n"
			"<</SYS>>\n\n"
			"Discharge Summary :\n"
			"{note}\n\n"
            #"Similar Patient Discharge Summary Context :\n"
            #"{context} \n\n"
			"Question : {question}\n"
			"Choices :\n"
			"A. {choice_a}\n"
			"B. {choice_b}\n"
			"C. {choice_c}\n"
			"D. {choice_d}\n"
			"E. {choice_e}\n\n"
			"Answer : "
			"[/INST]"
		)
	}
}


user = "Eric0929"

# Load the index
retriever = BM25HF.load_from_hub(f"{user}/bm25s-mimiciv", load_corpus=True)

def query(text):
    docs, scores = retriever.retrieve(bm25s.tokenize(text), k=2)
    return docs[0][1]['text']


def get_prompt(eval_method, model_name):
	print(eval_method, model_name)
	if "gpt" in model_name:
		return PROMPT_DICT[eval_method]["gpt"]
	elif model_name in ["meta-llama/Llama-2-70b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-7b-chat-hf"]:
		return PROMPT_DICT[eval_method]["llama-2-chat"]
