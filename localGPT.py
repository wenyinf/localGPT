#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)
import torch
import re

import pandas as pd
from pathlib import Path
import uuid
import csv
from datetime import datetime

from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY

# Hyperparameters
target_source_chunks = 1
source_path = "SOURCE_DOCUMENTS/"

def load_model(device_type, model_id, model_basename=None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    if device_type.lower() in ["cpu", "mps"]:
        model_basename = None

    if model_basename is not None:
        # The code supports all huggingface models that ends with GPTQ and have some variation
        # of .no-act.order or .safetensors in their HF repo.

        if ".safetensors" in model_basename:
            # Remove the ".safetensors" ending if present
            model_basename = model_basename.replace(".safetensors", "")

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        model = AutoGPTQForCausalLM.from_quantized(
            model_id,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0",
            use_triton=False,
            quantize_config=None,
        )
    elif (
        device_type.lower() == "cuda"
    ):  # The code supports all huggingface models that ends with -HF or which have a .bin
        # file in their HF repo.
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    else:
        #tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)
        #tokenizer.save_pretrained("tokenizer")
        #model.save_pretrained("model")
        tokenizer = LlamaTokenizer.from_pretrained("tokenizer")
        model = LlamaForCausalLM.from_pretrained("model")

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


def get_doc_source_path(db, query):
    pattern = r'[A-Za-z0-9 _]+\.csv'
    topic_doc = db.similarity_search(query, 
                                     k=1, 
                                     filter={"source": source_path + "Topic.csv"})
    match = re.search(pattern, topic_doc[0].page_content)
    file_name = match.group()
    return source_path + file_name.strip()

def get_qa(llm, db, query):
    """
    This function implements the information retrieval task.
    1. Get source path of most revelant document
    2. Loads the local LLM using load_model function - You can now set different LLMs.
    3. Setup the Question Answer retreival chain.
    """
    path = get_doc_source_path(db, query)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks, "filter": {"source": path}})

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

# Generates response to input given
def generate_answer(query, qa, return_source_documents=True):
    # Get the answer from the chain
    res = qa(query)
    date_str = datetime.now().strftime('%Y-%m-%d')
    if return_source_documents:
        answer, docs = res['result'], res['source_documents']
        return {"Answer": answer,
                "Document Source": [document.metadata["source"] for document in docs],
                "Document Content": [document.page_content for document in docs],
                "Date": date_str
            }
    else:
        answer = res['result']
        return {"Answer": answer,
                "Date": date_str
            }
    
def main(query):
    qa = get_qa(llm, db, query)
    generate_answer(query, qa)

if __name__ == "__main__":
    # Loads an embedding model, can be HuggingFaceInstructEmbeddings or HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Loads the existing vectorestore that was created by inget.py
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)

    # load the LLM for generating Natural Language responses

    # for HF models
    model_basename = None
    model_id = "TheBloke/guanaco-7B-HF"
    # model_id = 'NousResearch/Nous-Hermes-13b' # Requires ~ 23GB VRAM. Using STransformers
    # alongside will 100% create OOM on 24GB cards.

    # for GPTQ (quantized) models
    # model_id = "TheBloke/Nous-Hermes-13B-GPTQ"
    # model_basename = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"
    # model_id = "TheBloke/WizardLM-30B-Uncensored-GPTQ"
    # model_basename = "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors" # Requires
    # ~21GB VRAM. Using STransformers alongside can potentially create OOM on 24GB cards.
    # model_id = "TheBloke/wizardLM-7B-GPTQ"
    # model_basename = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"
    # model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"
    # model_basename = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"

    device_type = 'cuda'
    llm = load_model(device_type, model_id=model_id, model_basename=model_basename)
    query = 'what is the area project of Robot depalletizer & unscrambler of 18.5l drum for liquid filling area'
    main(query)