import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
from sentence_transformers import SentenceTransformer
import pickle
import time
import os 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#getitng the summary model and its tokenizer
if os.path.exists("t5_summary_model.pkl"):
    with open('t5_summary_model.pkl', 'rb') as f:
        summary_model = pickle.load(f)
    print("Summary model found in the disc, model is loaded successfully.")

else:
    print("Summary model does not exists in the path specified, downloading the model from web....")
    start_time = time.time()
    summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    end_time = time.time()

    print("downloaded the summary model in ",(end_time-start_time)/60," min , now saving it to disc...")

    with open("t5_summary_model.pkl", 'wb') as f:
        pickle.dump(summary_model,f)
    
    print("Done. Saved the model to disc.")

if os.path.exists("t5_summary_tokenizer.pkl"):
    with open('t5_summary_tokenizer.pkl', 'rb') as f:
        summary_tokenizer = pickle.load(f)
    print("Summary tokenizer found in the disc and is loaded successfully.")
else: 
    print("Summary tokenizer does not exists in the path specified, downloading the model from web....")

    start_time = time.time()
    summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    end_time = time.time()

    print("downloaded the summary tokenizer in ",(end_time-start_time)/60," min , now saving it to disc...")

    with open("t5_summary_tokenizer.pkl",'wb') as f:
        pickle.dump(summary_tokenizer,f)

    print("Done. Saved the tokenizer to disc.")


#Getting question model and tokenizer
if os.path.exists("t5_question_model.pkl"):
    with open('t5_question_model.pkl', 'rb') as f:
        question_model = pickle.load(f)
    print("Question model found in the disc, model is loaded successfully.")
else:
    print("Question model does not exists in the path specified, downloading the model from web....")
    start_time= time.time()
    question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
    end_time = time.time()

    print("downloaded the question model in ",(end_time-start_time)/60," min , now saving it to disc...")

    with open("t5_question_model.pkl", 'wb') as f:
        pickle.dump(question_model,f)
    
    print("Done. Saved the model to disc.")

if os.path.exists("t5_question_tokenizer.pkl"):
    with open('t5_question_tokenizer.pkl', 'rb') as f:
        question_tokenizer = pickle.load(f)
    print("Question tokenizer found in the disc, model is loaded successfully.")
else:
    print("Question tokenizer does not exists in the path specified, downloading the model from web....")

    start_time = time.time()
    question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
    end_time=time.time()

    print("downloaded the question tokenizer in ",(end_time-start_time)/60," min , now saving it to disc...")

    with open("t5_question_tokenizer.pkl",'wb') as f:
        pickle.dump(question_tokenizer,f)

    print("Done. Saved the tokenizer to disc.")

#Getting the sentence transformer model and its tokenizer
# paraphrase-distilroberta-base-v1
if os.path.exists("sentence_transformer_model.pkl"):
    with open("sentence_transformer_model.pkl",'rb') as f:
        sentence_transformer_model = pickle.load(f)
    print("Sentence transformer model found in the disc, model is loaded successfully.")
else:
    print("Sentence transformer model does not exists in the path specified, downloading the model from web....")
    start_time=time.time()
    sentence_transformer_model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v2")
    end_time=time.time()

    print("downloaded the sentence transformer in ",(end_time-start_time)/60," min , now saving it to disc...")

    with open("sentence_transformer_model.pkl",'wb') as f:
        pickle.dump(sentence_transformer_model,f)

    print("Done saving to disc.")
#Loading the models in to GPU if available
summary_model = summary_model.to(device)
question_model = question_model.to(device)