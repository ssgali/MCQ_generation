from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string
import pke
from textwrap3 import wrap
import random
import numpy as np
import nltk
nltk_data_dir = r"./"                                   # Current directory
nltk.data.path.append(nltk_data_dir)
import nltk_downloader                                  # For downloading the nltk files, this is one time, you can comment it once you have downloaded the files
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sense2vec import Sense2Vec
from similarity.normalized_levenshtein import NormalizedLevenshtein
from models_initiator import summary_model,question_model,summary_tokenizer,question_tokenizer,sentence_transformer_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We need to download the 2015 trained on reddit sense2vec model as it is shown to give better results than the 2019 one.
s2v = Sense2Vec().from_disk('s2v_old')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def postprocesstext(content):
    """
    this function takes a piece of text (content), tokenizes it into sentences, capitalizes the first letter of each sentence, and then concatenates the processed sentences into a single string, which is returned as the final result. The purpose of this function could be to format the input content by ensuring that each sentence starts with an uppercase letter.
    """
    final = ""
    for sent in sent_tokenize(content):
        sent = sent.capitalize()
        final = final + " " + sent
    return final


def summarizer(text, model, tokenizer):
    """
    This function takes the given text along with the model and tokenizer, which summarize the large text into useful information
    """
    text = text.strip().replace("\n", " ")
    text = "summarize: " + text
    # print (text)
    max_len = int(len(text.split()) * 0.5)
    encoding = tokenizer.encode_plus(
        text,
        max_length=max_len,
        pad_to_max_length=False,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        early_stopping=True,
        num_beams=8,
        num_return_sequences=5,
        no_repeat_ngram_size=4,
        min_length=75,
        max_length=300,
    )

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    summary = dec[0]
    summary = postprocesstext(summary)
    summary = summary.strip()

    return summary


def get_nouns_multipartite(content):
    """
    This function takes the content text given and then outputs the phrases which are build around the nouns , so that we can use them for context based distractors
    """
    out = []
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content, language="en")

        pos = {"PROPN", "NOUN", "ADJ", "VERB", "ADP", "ADV", "DET", "CONJ", "NUM", "PRON", "X"}
        stoplist = list(string.punctuation)
        stoplist += ["-lrb-", "-rrb-", "-lcb-", "-rcb-", "-lsb-", "-rsb-"]
        stoplist += stopwords.words("english")

        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(alpha=1.1, threshold=0.75, method="average")

        # Increase the number of keywords extracted
        keyphrases = extractor.get_n_best(no_of_mcqs=50)  # Increase n to 50 or more to generate more MCQs

        for val in keyphrases:
            out.append(val[0])
    except Exception as e:
        print(e)
        out = []

    return out


def get_keywords(originaltext):
    """
    This function takes the original text and the summary text and generates keywords from both which ever are more relevant
    This is done by checking the keywords generated from the original text to those generated from the summary, so that we get important ones
    """
    keywords = get_nouns_multipartite(originaltext)
    # print ("keywords unsummarized: ",keywords)
    # keyword_processor = KeywordProcessor()
    # for keyword in keywords:
    # keyword_processor.add_keyword(keyword)

    # keywords_found = keyword_processor.extract_keywords(summarytext)
    # keywords_found = list(set(keywords_found))
    # print ("keywords_found in summarized: ",keywords_found)

    # important_keywords =[]
    # for keyword in keywords:
    # if keyword in keywords_found:
    # important_keywords.append(keyword)

    # return important_keywords
    return keywords


def get_question(context, answer, model, tokenizer):
    """
    This function takes the input context text, pretrained model along with the tokenizer and the keyword and the answer and then generates the question from the large paragraph
    """
    text = "context: {} answer: {}".format(context, answer)
    encoding = tokenizer.encode_plus(
        text,
        max_length=384,
        pad_to_max_length=False,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        early_stopping=True,
        num_beams=8,
        num_return_sequences=5,
        no_repeat_ngram_size=4,
        max_length=72,
    )

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

    Question = dec[0].replace("question:", "")
    Question = Question.strip()
    return Question


def filter_same_sense_words(original, wordlist):
    """
    This is used to filter the words which are of same sense, where it takes the wordlist which has the sense of the word attached as the string along with the word itself.
    """
    filtered_words = []
    base_sense = original.split("|")[1]
    # print (base_sense)
    for eachword in wordlist:
        if eachword[0].split("|")[1] == base_sense:
            filtered_words.append(
                eachword[0].split("|")[0].replace("_", " ").title().strip()
            )
    return filtered_words


def get_highest_similarity_score(wordlist, wrd):
    """
    This function takes the given word along with the wordlist and then gives out the max-score which is the levenshtein distance for the wrong answers
    because we need the options which are very different from one another but relating to the same context.
    """
    score = []
    normalized_levenshtein = NormalizedLevenshtein()
    for each in wordlist:
        score.append(normalized_levenshtein.similarity(each.lower(), wrd.lower()))
    return max(score)


def sense2vec_get_words(word, s2v, topn, question):
    """
    This function takes the input word, sentence to vector model and top similar words and also the question
    Then it computes the sense of the given word
    then it gets the words which are of same sense but are most similar to the given word
    after that we we return the list of words which satisfy the above mentioned criteria
    """
    output = []
    # print ("word ",word)
    try:
        sense = s2v.get_best_sense(
            word,
            senses=[
                "NOUN",
                "PERSON",
                "PRODUCT",
                "LOC",
                "ORG",
                "EVENT",
                "NORP",
                "WORK OF ART",
                "FAC",
                "GPE",
                "NUM",
                "FACILITY",
            ],
        )
        most_similar = s2v.most_similar(sense, n=topn)
        # print (most_similar)
        output = filter_same_sense_words(sense, most_similar)
        # print ("Similar ",output)
    except:
        output = []

    threshold = 0.6
    final = [word]
    checklist = question.split()
    for x in output:
        if (
            get_highest_similarity_score(final, x) < threshold
            and x not in final
            and x not in checklist
        ):
            final.append(x)

    return final[1:]


def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):
    """
    The mmr function takes document and word embeddings, along with other parameters, and uses the Maximal Marginal Relevance (MMR) algorithm to extract a specified number of keywords/keyphrases from the document. The MMR algorithm balances the relevance of keywords with their diversity, helping to select keywords that are both informative and distinct from each other.
    """

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphrase
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(
            word_similarity[candidates_idx][:, keywords_idx], axis=1
        )

        # Calculate MMR
        mmr = (lambda_param) * candidate_similarities - (
            1 - lambda_param
        ) * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]


def get_distractors_wordnet(word):
    """
    the get_distractors_wordnet function uses WordNet to find a relevant synset for the input word and then generates distractor words by looking at hyponyms of the hypernym associated with the input word. These distractors are alternative words related to the input word and can be used, for example, in educational or language-related applications to provide choices for a given word.
    """
    distractors = []
    try:
        syn = wn.synsets(word, "n")[0]

        word = word.lower()
        orig_word = word
        if len(word.split()) > 0:
            word = word.replace(" ", "_")
        hypernym = syn.hypernyms()
        if len(hypernym) == 0:
            return distractors
        for item in hypernym[0].hyponyms():
            name = item.lemmas()[0].name()
            # print ("name ",name, " word",orig_word)
            if name == orig_word:
                continue
            name = name.replace("_", " ")
            name = " ".join(w.capitalize() for w in name.split())
            if name is not None and name not in distractors:
                distractors.append(name)
    except:
        print("Wordnet distractors not found")
    return distractors


def get_distractors(
    word, origsentence, sense2vecmodel, sentencemodel, top_n, lambdaval
):
    """
    this function generates distractor words (answer choices) for a given target word in the context of a provided sentence. It selects distractors based on their similarity to the target word's context and ensures that the target word itself is not included among the distractors. This function is useful for creating multiple-choice questions or answer options in natural language processing tasks.
    """
    distractors = sense2vec_get_words(word, sense2vecmodel, top_n, origsentence)
    # print ("distractors ",distractors)
    if len(distractors) == 0:
        return distractors
    distractors_new = [word.capitalize()]
    distractors_new.extend(distractors)
    # print ("distractors_new .. ",distractors_new)

    embedding_sentence = origsentence + " " + word.capitalize()
    # embedding_sentence = word
    keyword_embedding = sentencemodel.encode([embedding_sentence])
    distractor_embeddings = sentencemodel.encode(distractors_new)

    # filtered_keywords = mmr(keyword_embedding, distractor_embeddings,distractors,4,0.7)
    max_keywords = min(len(distractors_new), 5)
    filtered_keywords = mmr(
        keyword_embedding,
        distractor_embeddings,
        distractors_new,
        max_keywords,
        lambdaval,
    )
    # filtered_keywords = filtered_keywords[1:]
    final = [word.capitalize()]
    for wrd in filtered_keywords:
        if wrd.lower() != word.lower():
            final.append(wrd.capitalize())
    final = final[1:]
    return final


def get_mca_questions(context: str):
    """
    this function generates multiple-choice questions based on a given context. It summarizes the context,
    extracts important keywords, generates questions related to those keywords, and provides randomized
    answer choices, including the correct answer, for each question.
    """
    summarized_text = summarizer(context, summary_model, summary_tokenizer)

    # imp_keywords = get_keywords(context ,summarized_text)
    imp_keywords = get_keywords(context)
    output_list = []
    for answer in imp_keywords:
        output = ""
        ques = get_question(summarized_text, answer, question_model, question_tokenizer)

        distractors = get_distractors(
            answer.capitalize(), ques, s2v, sentence_transformer_model, 40, 0.2
        )

        output = output + ques + "\n"
        if len(distractors) == 0:
            distractors = imp_keywords

        if len(distractors) > 0:
            random_integer = random.randint(0, 3)
            alpha_list = ["(a)", "(b)", "(c)", "(d)"]
            for d, distractor in enumerate(distractors[:4]):
                if d == random_integer:
                    output = output + alpha_list[d] + answer + "\n"
                else:
                    output = output + alpha_list[d] + distractor + "\n"
            output = (
                output + "Correct answer is : " + alpha_list[random_integer] + "\n\n"
            )

        output_list.append(output)

    mca_questions = output_list
    return mca_questions
