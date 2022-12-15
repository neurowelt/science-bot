# Science ChatBot

Are you looking for answers to your scientific questions? This guy can answer some of them.

## Description

This repository is an attempt to create a ChatBot answering scientific questions. It implements several ideas that scratch the surface of building a reliable, functional and trustworthy ChatBot.

ChatBot looks for an answer to the question by:
1. Querying its local knowledge base to find the most similar piece of information
2. Querying Wikipedia and Google to do the same

In both cases the so called **context** is retrieved. The assumption is that context contains an answer to the question. For example:

```python
context = "John was a young man having a great time during his time studying at University of Warsaw."
question = "Where did John study?"
answer = "University of Warsaw"
```

ChatBot uses BERT model, which tries to find an answer based on a given question and context.

## Installation

There are two steps necessary to launch the ChatBot.

### Python

To run the bot `python>=3.9` is required (older versions were not tested). If you have Python installed, you can just install packages manually using the `requirements.txt` file.

```shell
> pip install -r requirements.txt
```

Since [transformers](https://github.com/huggingface/transformers) library is handling installation of language models, they are installed with the first launch of the bot and do not have to be mentioned in this part.

If you are an Apple M1 user, you will find some useful information [here](#Installing-on-MacOS-with-M1).

**NOTE**<br>
Installation is currently limited to my own experiences, therefore:
1. It was not tested on Windows
2. For Mac users, installation by default creates an Intel-based environment (installs for x86 architecture)

Hope it's not a trouble!

### Datasets

In order to construct a local knowledge base for the ChatBot two open source datasets were used:
* [SciQ](https://allenai.org/data/sciq)
* [COQA](https://paperswithcode.com/dataset/coqa)

Based on them, three data files with knowledge base were prepared (using the `knowledge.py` script) and can be downloaded using `datasets.sh` script.

Run the following script to download the knowledge base:

```shell
> chmod +x dataset.sh
> ./dataset.sh
```

The `+x` argument makes the script executionable.

## Models

The following models were used in this project:
* Knowledge base model: `bert-large-uncased-whole-word-masking-finetuned-squad`
* Encoding model: `bert-large-uncased-whole-word-masking-finetuned-squad`
* NER fine-tuned BERT model: `Jorgeutd/bert-large-uncased-finetuned-ner`
* SpaCy model: `en_core_web_sm`

## Methods

In the current version of the project four main ideas came to fruition:
* Knowledge base with entity tagging and similarity search
* Wikipedia querying for summaries
* Noun phrases extraction for entity recognition support
* Google Quick Info box

### Knowledge base

ChatBot when asked a question tries to find the closest question and context assigned to it that it has in its knowledge base. In order to do it the following steps are performed:
1. Find *entities* in the question (i.e. important objects like person, location, organization)
2. Encode the question (i.e. represent it in a form of BERT latent space embeddings)
3. Find a subset of knowledge base with recognized entities
4. Find the closest question in the knowledge base to the input question (i.e. check encoding similarity)
5. Return the context assigned to the most similar question

Knowledge base consists of three different datasets:
1. `knowledge_base.pkl`: dicitonary with each entry of the following structure:<br>
    `{ UNIQUE_ID : [ CONTEXT , QUESTION , ANSWER ] }`<br>
    Each element is derived as follows:
    * `unique_id`: this key is built upon iterating over questions, so the ID essentially represents question number
    * `context`: depending on the dataset, this is the description of the problem (`COQA`) or the elaborated answer (`SciQ`)
    * `question`: the questions concerning given context
    * `answer`: answers to the questions (finally not used in the ChatBot process)
2. `entity_base.pkl`: dictionary containing all the entities that were detected in the contexts found in the `knowledge_base`
3. `encoding_base_comp.npz`: numpy array filled with encodings (ordered in the same way as IDs in `knowledge_base`)

### Querying Wikipedia

Truth be told, Wikipedia is **the** source of knowledge in the Internet. ChatBot takes advantage of that using [wikipedia](https://github.com/goldsmith/Wikipedia) library to call to query its API looking for answers.

The main assumption of this approach is to retrieve the first couple of sentences and pass them as context to the BERT model.

### Noun phrases

Models pre-trained on NER (Named-Entity Recognition) tasks are aware of specific entities, while general entities are omitted (as stated in the name of the task). For example:

```python
entities = NER("How many legs a human has?")
print(entities)
> []
```

While we know that `human` is an entity, it didn't ring a bell for the pre-trained model (this result is of course **based on my experience with models used in this project**).

Solution to that is a linguistic approach - noun chunks. Using [spacy](https://github.com/explosion/spaCy) built-in sentence parsing we can extract all the noun chunks - meaningful phrases that represent **nouns** and all other words related to them. That way we extract `human` as one of such chunks and it can be passed further to the ChatBot pipeline as an entity!

### Google Quick Info

When you enter a question in Google Search, you often get something like this:

![Google Quick Info example](./images/google-quick-info.png)

ChatBot queries a question and tries to retrieve an answer stored in this particular box. If it succeeds, it is passed as context.

## Additional information

### Installing on MacOS with M1

In order to prepare the environment you can use the `install.sh` script (installer uses [conda](https://github.com/conda-forge/miniforge), so using it requires this package to be installed).

Running this script will create an environment called `chatbot` with all the packages required, as well as download the knowledge base (basically run `datasets.sh` script).

**IMPORTANT**<br>
In order to run it follow the commands below (as with the `dataset.sh`):

```shell
> chmod +x pystuff.sh
> ./pystuff.sh
```

The `+x` argument makes the script executionable.