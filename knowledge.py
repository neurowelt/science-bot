# Basic
import json, os, pickle
import numpy as np

# Models
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import BertForTokenClassification, BertTokenizer
from transformers import pipeline


def parse_json(filepath):

    '''
    Description
    -----------
        Simple function loading .json files.
    '''

    if ".json" in filepath:
        with open(filepath, "r") as f:
            return json.loads(f.read())
    else:
        return None


def extract_knowledge(file_dir, dataset, counter=0):

    '''
    Description
    -----------
        Function extracting information from .json datasets. For now it supports only two datasets:
        *   SciQ dataset (link)
        *   COQA Stanford dataset (link)
        
        Each is parsed in a way to create the structure as follows:
            { UNIQUE_ID : [ CONTEXT , QUESTION , ANSWER ] }

        Context is used as the background from which the answer to question can be inferred. It allows
        models like BERT perform Question Answering tasks.
    '''

    knowledge_base = {}
    fls = os.listdir(file_dir)
    
    if dataset == "sciq":
        for fl in fls:
            parsed_j = parse_json(os.path.join(file_dir, fl))
            if parsed_j:
                for el in parsed_j:
                    knowledge_base[counter] = [str(el["support"]), el["question"], str(el["correct_answer"]).capitalize()]
                    counter += 1
    
    elif dataset == "coqa":
        for fl in fls:
            parsed_j = parse_json(os.path.join(file_dir, fl))
            data = parsed_j["data"] if parsed_j else []
            for el in data:
                for _, z in enumerate(zip(el["questions"], el["answers"])):
                    knowledge_base[counter] = [el["story"], z[0]["input_text"], z[1]["span_text"]]
                    counter += 1


    return knowledge_base, counter


def create_knowledge_base(source, dataset, enc_model, ent_model, save=True):

    '''
    Description
    -----------
        Main function for creating knowledge base. The following bases are constructed:
        *   knowledge_base - contains information about context, questions and answers extracted
            from datasets specified in source+dataset arguments
        *   encoding_base - contains embeddings of sentences in specified language model
        *   enitty_base - contains information about entity occurences in given contexts

        It is worth noting, that for the datasets used in this project, time for performing
        NER task on each context may take a lot of time. Only after performing the whole
        extraction process I realized, that it is performed many times for the same contexts, which
        should be fixed later on.
    '''

    if isinstance(source, str):
        source = [source]
    if isinstance(dataset, str):
        dataset = [dataset]
    assert len(source) == len(dataset), "Mismatching lengths of source and dataset arguments"
    device = 0 if torch.cuda.is_available() else -1

    enc_model = SentenceTransformer(enc_model)
    ent_model = BertForTokenClassification.from_pretrained(ent_model)
    #tokenizer = BertTokenizer.from_pretrained(str(ent_model)) #- HFValidationError thrown when passing string as argument, didn't manage to find the solution
    tokenizer = BertTokenizer.from_pretrained("Jorgeutd/bert-large-uncased-finetuned-ner")
    ner = pipeline("ner", model=ent_model, tokenizer=tokenizer, device=device) #- on M1 Max showed ~9h for coqa+sciq, used A100 on Paperspace to get it to ~1h

    knowledge_base = {}
    id_count = 0
    print("Extracting information from .json datasets...")
    for i, s in enumerate(source):
        assert os.path.isdir(s), print("Pass only directories with .json files, not specific filepaths.")
        print(f"\tNow extracting: {dataset[i]}")
        kb, counter = extract_knowledge(file_dir=s, dataset=dataset[i], counter=id_count)
        knowledge_base = {**knowledge_base, **kb}
        id_count += counter

    encoding_base = []
    entity_base = {}
    print("Encoding questions and performing entity recognition...")
    for i in tqdm(range(len(knowledge_base.keys()))):
        k = list(knowledge_base.keys())[i]
        q = str(knowledge_base[k][1]) #- we know the position of a question
        encoding_base.append(enc_model.encode(q))
        
        s = str(knowledge_base[k][0]) #- and the position of context
        entities = ner(s)
        entity_key = []
        for j, ent in enumerate(entities):
            if "B-" in ent["entity"] and len(entity_key) >= 1:
                en_name = " ".join(entity_key)
                entity_key = []
            
                if en_name not in entity_base.keys():
                    entity_base[en_name] = [i]
                else:
                    entity_base[en_name].append(i)

            entity_key.append(ent["word"])

        if len(entity_key) > 0:
            en_name = " ".join(entity_key)
            entities[en_name] = ent["entity"].split("-")[-1]

    print("Saving...")
    if save:
        with open('knowledge_base.pkl', 'wb') as f:
            pickle.dump(knowledge_base, f)

        np.save('encoding_base.npy', np.array(encoding_base))

        with open('entity_base.pkl', 'wb') as f:
            pickle.dump(entity_base, f)

    print("Finished creating knowledge base!\n"\
        f"Number of questions: {id_count}\n"\
        f"Number of entites: {len(entity_base.keys())}\n"\
        "All files can be found in the working directory."
        )

    return knowledge_base, encoding_base, entity_base


if __name__ == "__main__":

    create_knowledge_base(
        source=["./sciq", "."],
        dataset=["sciq", "coqa"],
        enc_model="bert-large-uncased-whole-word-masking-finetuned-squad",
        ent_model="Jorgeutd/bert-large-uncased-finetuned-ner",
        save=True
    )