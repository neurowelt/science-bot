# Basic
import argparse, itertools, os, pickle, re, requests
import numpy as np

# Query
import bs4
import wikipedia as wp
from wikipedia import PageError
from googlesearch import search
from urllib.parse import urlencode

# Models
import torch
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from transformers import BertForTokenClassification, BertTokenizer, BertForQuestionAnswering


def get_entities(entity_list, filter=None):

    '''
    Description
    -----------
        Retrieve entity type and name from BERT Token Classifier. Allows for filtering by entity type.
        Pass entity types according to NER task that you would like to extract. CHUNK type was added to
        allow using this function for noun chunk extraction.
    '''

    ENT_TYPES = ["O", "MIS", "LOC", "ORG", "PER", "CHUNK"]
    if isinstance(filter, str):
        filter = [filter]
    if filter:
        assert np.all([x in ENT_TYPES for x in filter]), f"All filters must be one of these: {ENT_TYPES}"
    else:
        filter = ENT_TYPES

    entity_key = []
    entities = {}
    for _, ent in enumerate(entity_list):
        if ent["entity"].split("-")[-1] in filter:
            if "B-" in ent["entity"] and len(entity_key) >= 1:
                en_name = " ".join(entity_key)
                entity_key = []
                entities[en_name] = ent["entity"].split("-")[-1]

            entity_key.append(ent["word"])
    
    if len(entity_key) > 0:
        en_name = " ".join(entity_key)
        entities[en_name] = ent["entity"].split("-")[-1]
    
    return entities


def cos_similarity(a, b):

    '''
    Description
    -----------
        Calculate cosine similarity for two vectors.
    '''

    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


class ChatBot:

    '''
    Description
    -----------
        Class for constructing the ChatBot.
    '''

    def __init__(
        self,
        answer_model="bert-large-uncased-whole-word-masking-finetuned-squad",
        entity_model="Jorgeutd/bert-large-uncased-finetuned-ner",
        encoder_model="bert-large-uncased-whole-word-masking-finetuned-squad",
        spacy_model="en_core_web_sm",
        knowledge_base="./data/knowledge_base.pkl",
        entity_base="./data/entity_base.pkl",
        encoding_base="./data/encoding_base_comp.npz",
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 OPR/93.0.0.0"
    ):
        
        print("Initializing models...")
        # BERT model trained to perform Question Answering tasks
        self.answer_model = BertForQuestionAnswering.from_pretrained(answer_model)
        self.answer_tokenizer = BertTokenizer.from_pretrained(answer_model)
        
        # BERT model trained to perform Entity Recognition tasks
        entity_model = BertForTokenClassification.from_pretrained(entity_model)
        try: 
            entity_tokenizer = BertTokenizer.from_pretrained(entity_model) #- HFValidationError thrown when passing string as argument, didn't manage to find the solution
        except Exception as e:
            entity_tokenizer = BertTokenizer.from_pretrained("Jorgeutd/bert-large-uncased-finetuned-ner")
        self.ner = pipeline("ner", model=entity_model, tokenizer=entity_tokenizer)

        # Model used to get sentence embeddings from BERT
        self.encoder = SentenceTransformer(encoder_model)

        # If entity is not recognized, try finding subject with spacy
        self.subject_nlp = spacy.load(spacy_model)

        print("Loading knowledge base...")
        with open(knowledge_base, "rb") as f:
            self.knowledge_base = pickle.load(f)

        print("Loading entity base...")
        with open(entity_base, "rb") as f:
            self.entity_base = pickle.load(f)

        print("Loading encoding base...")
        self.encoding_base = np.load(encoding_base, allow_pickle=True)["arr_0"]

        # User agent for querying Google
        self.user_agent = user_agent

        # Helpful counter and status
        self.bad_answers = 0
        self.status = 1


    def process_question(self, question, **kwargs):

        '''
        Description
        -----------
            Retrieve entities and encode the question asked for improving context search later.
        '''
        
        ner_res = self.ner(question)
        entities = get_entities(ner_res, **kwargs) if len(ner_res) > 0 else []
        q_encoding = self.encoder.encode(question)

        return entities, q_encoding


    def question_search(self, q_encoding, subset=None):

        '''
        Description
        -----------
            Search the encodings base to find context for the most similar question to the one user asked. 
            If subset of ids is passed (for example from entity_base) the search is quickly limited to the 
            questions concerning texts with entities found in the question asked.
        '''

        encoding_base = self.encoding_base[np.array(subset, dtype=np.uint16)] if subset is not None else self.encoding_base
        knowledge_pool = {i: self.knowledge_base[k] for i,k in enumerate(subset)} if subset is not None else self.knowledge_base #- subset dict for index to match
        res = np.apply_along_axis(cos_similarity, -1, encoding_base, q_encoding)
        most_sim_ctx = knowledge_pool[np.argmax(res)][0][:512] #- we know index from the structure and max_cap for the default model

        return most_sim_ctx


    def wiki_query(self, entities, summ_res=0):
        
        '''
        Description
        -----------
            Perform query to Wikipedia API to retrieve summary for given entities. You can specify wich query from the list to summarize.
            For now the behaviour is limited to the queries that match regex rules.
        '''

        if isinstance(entities, dict):
            entities = list(entities.keys())
        reg_con = "&".join(entities)
        reg_alt = "|".join(entities)
        query = " ".join(entities)
        res = [t.lower().replace("?", "") for t in wp.search(query, results=10)]
        if len(res) == 0:
            return res

        checks = [bool(re.search(reg_con, q)) for q in res]
        if sum(checks) == 0:
            checks = [bool(re.search(reg_alt, q)) for q in res] #- if conjunction is too strong, try alternative

        try:
            if sum(checks) == 0:
                return [wp.summary(res[0], sentences=5)] #- neither entity found in the title, let's count it's the first one
            else:
                idx = np.arange(0,10)
                titles = np.array(res)[idx[checks]]
                
                # If we reached the last summary found, just return the first one
                if self.bad_answers == len(titles): print("No more new matching summaries. Returning to first one.")
                summ_res = min(summ_res, len(titles)-1) if isinstance(summ_res, int) and summ_res > 0 else 0
                
                return [wp.summary(titles[summ_res], sentences=5)]
        except PageError:
            return []


    def process_answer(self, question, context):

        '''
        Description
        -----------
            Process question and context in order to use BERT Question Answering functionality.
        '''

        # Encode question and context
        input_ids = self.answer_tokenizer.encode(question, context)
        sep_pos = input_ids.index(self.answer_tokenizer.sep_token_id)
        len_a = sep_pos + 1
        segment_ids = [0]*len_a + [1]*(len(input_ids)-len_a)
        output = self.answer_model(
            input_ids=torch.tensor([input_ids]),
            token_type_ids=torch.tensor([segment_ids])
        )
        tokens = self.answer_tokenizer.convert_ids_to_tokens(input_ids)

        # Get our answer 
        answer_start = torch.argmax(output.start_logits)
        answer_end = torch.argmax(output.end_logits)
        if answer_end >= answer_start:
            answer = " ".join(tokens[answer_start:answer_end+1]).replace(" ##", "")
            answer.replace("##","")
        else:
            answer = None

        return answer


    def search_google(self, query, user_agent, exp_mode=False):

        '''
        Description
        -----------
            Search the question in Google in order to retrieve the title of Wikipedia article that may answers it.
            It can be used in an experimental mode, which extracts the context. This mode is described in detail in README file.
        '''

        if not exp_mode:
            
            gsearch = list(search(query,  tld='com', lang='en', tbs='0', safe='off', stop=5))
            reg = r"(\S)*wikipedia(\S)*"
            res = [bool(re.search(reg, r)) for r in gsearch]
            wiki_page = gsearch[np.argmax(res*1)]

            return wiki_page.split("/")[-1].replace("_", " ")

        else:
            
            headers = {
                'User-Agent': user_agent,
                'Accept-Language': 'en-US,en;q=0.5'
                }

            q = {
                'q': query
            }

            uniq_h = {
                'h_en': 'Featured snippet from the web',
                'h_pl': 'Fragment z odpowiedzią z internetu'
            }

            gquery = f"https://www.google.com/search?{urlencode(q)}"

            html = requests.get(gquery, headers=headers)
            soup = bs4.BeautifulSoup(html.content, 'lxml')

            # Attempt to find the quickinfo_id
            str_soup = str(soup)
            h_len = -1
            start = -1
            quickinfo_id = None
            for _, h in enumerate(uniq_h):

                # Find unique header
                head = uniq_h[h]
                reg_head = f'>{head}<'
                span = re.search(reg_head, str_soup)

                # If you fuond it, find the class id it belongs to
                if span:
                    reg_div = r'<div class=(.)*>'
                    stop, _ = span.span()
                    start = stop - 53
                    h_len = len(head)
                    try:
                        quickinfo_id = re.search(reg_div, str_soup[start:stop]).group()
                    except AttributeError:
                        quickinfo_id = None

            if quickinfo_id:
                quickinfo_id = quickinfo_id[1:-1].split("=")[-1][1:-1]
                res = soup.find("div", class_=quickinfo_id)
                context = res.text[h_len:].split("›")[0]
            else:
                context = None

            return context


    def get_context(self, question, web=False, google_support=False, find_chunks=True, reverse_web=False, **kwargs):

        '''
        Descriptions
        -----------
            Get context for question answering by either:
            *   Searching through the knowledge base (default behavior)
            *   Querying wiki for a short topic summary
            Additional arguments added for debugging and research purposes are:
            *   google_support - Enables Google-based search that points to adequate Wikipedia article (can be set True or is used when no entities are found)
            *   find_chunks - if there are no entities found, try using spacy dependency tree to find noun chunks
            *   reverse_wiki - reverse the behavior of this function where first option to check is Wikipedia query
        '''

        # Use wiki as primary method
        web = not web if reverse_web else web

        # Prepare entities and encoding
        filter = kwargs["filter"] if "filter" in kwargs.keys() else None
        entities, q_encoding = self.process_question(question, filter=filter)

        # If no entities were found, try finding noun chunks
        if len(entities) < 1 and find_chunks:
            syntax_tree = self.subject_nlp(question)
            entities = [{"entity": "B-CHUNK", "word":tok.text.lower()} for tok in syntax_tree.noun_chunks]
            entities = get_entities(entities, filter=["CHUNK"])
            web = True #- force wiki, as knowledge base was done using NER, and if we get to chunks, they are not in entity base

        if web is False:

            # Limit searching through knowledge base if any entity was found in the question
            if len(entities) > 0:
                entity_pool = [self.entity_base[ent] for ent in entities.keys() if ent in entities.keys()]
                if len(entity_pool) < 1:
                    entity_pool = None
                else:
                    entity_pool = [entity_pool] if not isinstance(entity_pool[0], list) else entity_pool
                    entity_pool = np.array(list(set(itertools.chain(*entity_pool))), dtype=np.uint16) #- get rid of duplicates and cast to np for indexing
            else:
                entity_pool = None #- search the whole base
            
            context = self.question_search(q_encoding, subset=entity_pool)

            return context

        elif web is True:
            exp_mode = kwargs["exp_mode"] if "exp_mode" in kwargs.keys() else False

            # If we failed at experimental option, fall back to the safer method
            while True:
                if not exp_mode:

                    # If no entities or chunks were found, find the page on Wiki, using the help of Google
                    # Also can be enabled using google_support=True argument
                    if len(entities) < 1 or google_support:
                        entities = self.search_google(question, self.user_agent)

                    summ_res = kwargs["summ_res"] if "summ_res" in kwargs.keys() else 0
                    context = self.wiki_query(entities=entities, summ_res=summ_res)
                    context = None if len(context) == 0 else context[0]

                else:

                    context = self.search_google(question, self.user_agent, exp_mode=exp_mode)
                    if context is None:
                        exp_mode = False
                        context = -1

                if context == -1:
                    continue
                else:
                    break

            return context

        else:
            raise TypeError("Argument web can only be of type boolean.")


    def answer_question(self, question, **kwargs):

        '''
        Description
        -----------
            Answer question given based on the context inferred from it. Allows for passing keyword arguments to get_context,
            specifically to perform Wiki query if knowledge-based inferring did not work out.
        '''

        context = self.get_context(question, **kwargs)

        if context:
            answer = self.process_answer(question, context)

            if answer: return answer
        
        self.bad_answers += 1
        return


    def save_answers(self, questions, answers):

        '''
        Description
        -----------
            Save all questions with answers into a .txt file.
        '''

        with open("./answers.txt", "w") as f:
            for i, q in enumerate(questions):
                f.write(f"Question {i}:\n{q}\n\nAnswer:\n{answers[i]}\n\n\n")

    
    def launch(self, q_src=".", interactive=True, bad_ans_thresh=2, **kwargs):

        '''
        Description
        -----------
            Launch the bot. Use argument interactive=True in order to enable friendly responses and possibility to find better answer.
            If interactive=False the file with questions need to be specified (simple .txt file). At the end, all questions and answered
            are saved into a .txt file. You can manage how long to look for an answer when there is nothing that bot can finds by adjusting
            the bad_ans_thresh.
        '''
        
        questions = []
        if os.path.isfile(q_src) and interactive is False:
            with open(q_src, "r") as f:
                questions = [line.strip() for line in f.readlines() if line != "\n"]
        elif interactive is True:
            question = input("Your question:\n")
            questions.append(question)
        else:
            print("Problem with launching ocurred. Make sure you passed the correct path to questions file or activated the interactive mode.")
            return

        interactive *= 1
        self.status *= interactive
        web = True
        summ_res = 0 #- used by wiki_query
        
        answers = []
        _questions = [] #- for non-interactive
        while True:
            
            if len(questions) == 0:
                self.save_answers(_questions, answers)
                break

            answer = self.answer_question(questions[0], web=web, summ_res=summ_res, filter=["PER", "LOC", "ORG"], **kwargs)

            if answer:
                web = False #- return to normal behavior
                self.bad_answers = 0

                # Try querying wiki for better context if user says so
                if self.status != 0:
                    print(f"\nAnswer:\n{answer}\n")
                    correct = input("Are you satisfied with the answer? [y/n]")
                    if correct.lower()[0] == "n":
                        
                        print("Just a moment please, I'm contacting the oracle...")

                        summ_res = self.bad_answers
                        self.bad_answers += 1
                        web = True #- user requested further search

                    # Answer was correct
                    else:
                        answers.append(answer)
                        
                        next = input("Great! Is there anything else? [y/n]")
                        if next.lower()[0] == "n":
                            print("Thank you and see you later!")
                            self.save_answers(questions, answers)
                            return
                        
                        # User wants to ask more questions
                        else:
                            question = input("Your question:\n")
                            questions.insert(0, question)

                # No interactivity - just move to the next question
                else:
                    answers.append(answer)
                    _questions.append(questions.pop(0))

            else:
                if self.status != 0:
                    print("Sorry, got a bit confused, give me just a little moment...")

                summ_res = self.bad_answers
                self.bad_answers += 1
                web = True

                if self.bad_answers > bad_ans_thresh:
                    answers.append("Unfortunately, no answer was found. Please try to rephrase the question.")
                    web = False
                    self.bad_answers = 0
                        
                    # Give option to stop chatting at this point too
                    if self.status != 0:
                        
                        question = input("I couldn't the answer. I'm really sorry. Please, enter a different question or input [n] to stop:\n")
                        if question.lower() == "n":
                            print("Thank you and see you later!")
                            self.save_answers(questions, answers)
                            return
                        
                        # User wants to ask more questions
                        else:
                            questions.insert(0, question)

                    # Shorter non-interactive way
                    else:
                        _questions.append(questions.pop(0))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", default="./questions.txt", help="Path to the file with questions.", type=str)
    parser.add_argument("-i", "--interactive", default="store_true", help="If True, the bot will launch in an interactive form.", type=bool)
    parser.add_argument("-e", "--exp_mode", default="store_false", help="Experimental mode using Google Quick Info as context base.", type=bool)

    args = parser.parse_args()
    bot = ChatBot()
    bot.launch(
        q_src=args.source,
        interactive=False,
        exp_mode=True,
    )