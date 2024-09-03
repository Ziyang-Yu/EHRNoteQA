import bm25s
from bm25s.hf import BM25HF

PROMPT_DICT = {
	"openended": {
		"gpt": (
			"Discharge Summary :\n"
			"{note}\n\n"
                        "Context :\n"
                        "{context}\n\n"
			"Question : {question}\n\n"
			"Answer :"
        ),
		"llama-2-chat": (
			"[INST] <<SYS>>\n"
			"You are a helpful, respectful and honest assistant.\n"
			"<</SYS>>\n\n"
			"Discharge Summary :\n"
			"{note}\n\n"
                        "Context :\n"
                        "{context}\n\n"
			"Question : {question}\n\n"
			"Answer : "
			"[/INST]"
        ),
	},
	"multichoice": {
		"gpt": (
			"Discharge Summary :\n"
			"{note}\n\n"
                        "Context :\n"
                        "{context}\n\n"
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
                        "Context :\n"
                        "{context}\n\n"
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


def get_prompt(eval_method, model_name):
	print(eval_method, model_name)
	if "gpt" in model_name:
		return PROMPT_DICT[eval_method]["gpt"]
	elif model_name in ["Llama-2-70b-chat-hf", "Llama-2-13b-chat-hf", "Llama-2-7b-chat-hf"]:
		return PROMPT_DICT[eval_method]["llama-2-chat"]

retriever = BM25HF.load_from_hub(f"Eric0929/bm25s-mimiciv", load_corpus=True)
def get_context(query, model='BM25s'):

        if model == 'BM25s':

            # Load the index
            #retriever = BM25HF.load_from_hub(f"{user}/bm25s-mimiciv", load_corpus=True)

            # You can retrieve now
            docs, scores = retriever.retrieve(bm25s.tokenize(query), k=2)

            return docs[0][1]['text']

        else:
            raise NotImplementedError
