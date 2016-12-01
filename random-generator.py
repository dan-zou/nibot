import pandas as pd
import nltk
import numpy as np

from nltk.tag.perceptron import PerceptronTagger

import datetime
import re

chat_history_directory = "FILL ME IN"

def tokenize_and_tag(series):
    
    tagger = PerceptronTagger()
    return series.map(lambda x: tagger.tag(nltk.word_tokenize(x)))

def parallel_tokenize(series):
    
    num_cores = 8
    series_split = np.array_split(series, num_cores)
    pool = Pool(num_cores)
    result = pd.concat(pool.map(tokenize_and_tag, series_split))
    
    pool.close()
    pool.join()
    
    return result

from multiprocessing import Pool

rooms = ['CHAT ROOM 1','CHAT ROOM 2']

result_dfs = []

for room in rooms:
    
    print "(" + str(datetime.datetime.now()) + ") Room: " + room
    room_jsons = pd.read_json(chat_history_directory + "rooms/" + room + "/history.json")
    room_messages = room_jsons[pd.notnull(room_jsons['UserMessage'])]['UserMessage'].apply(pd.Series)
    room_messages['date'] = room_messages['timestamp'].map(lambda x: pd.to_datetime(x[0:10]))
    room_messages['time'] = room_messages['timestamp'].map(lambda x: x[11:19])
    room_senders = room_messages['sender'].apply(pd.Series)
    
    room_df = room_messages[['date','time','message']]
    room_df['user'] = room_senders['name']
    
    room_df = room_df[room_df['message'].map(lambda x: len(x) > 0)]
    room_df = room_df[room_df['message'].map(lambda x: (x[0] != 's') and (x[1] != "/") if len(x) > 1 else True)]
    
    room_df['emojis'] = room_df['message'].map(lambda x: re.findall("\([A-Za-z][a-z]+\)",x))
    room_df['cleaned_message'] = room_df['message'].map(lambda x: re.sub("\([^)]*\)",' Emoji ',x))
    
    room_df['callouts'] = room_df['cleaned_message'].map(lambda x: re.findall("@[^ ]+", x))
    room_df['cleaned_message'] = room_df['cleaned_message'].map(lambda x: re.sub("@[^ ]+", " Callout ",x))
    
    room_df['websites'] = room_df['cleaned_message'].map(lambda x: re.findall("http[^ ]+", x))
    room_df['cleaned_message'] = room_df['cleaned_message'].map(lambda x: re.sub("http[^ ]+", " Website ", x))
    
    room_df = room_df[room_df['cleaned_message'].map(lambda x: len(x.strip()) > 0)]
    room_df['tokens'] = parallel_tokenize(room_df['cleaned_message'])
    
    # room_df = room_df[room_df['time'].map(lambda x: str(x)[0:2] in ["11","12"])]
    result_dfs.append(room_df)
    
    print "(" + str(datetime.datetime.now()) + ") Finished!"

result_df = pd.concat(result_dfs)


# construct pos corpus
pos_corpus_df = pd.concat(list(result_df['tokens'].map(lambda x: pd.DataFrame(x))))
pos_corpus_df.columns = ['word','pos']
pos_corpus_df['word'] = pos_corpus_df['word'].map(lambda x: [x])
pos_corpus_df = pos_corpus_df.groupby('pos').sum()
pos_corpus_df['word'] = pos_corpus_df['word'].map(lambda x: [y for y in x if (y not in ['Website']) and len(y) > 2])
pos_corpus = pos_corpus_df.T.to_dict()

# create sentence structure corpus
sentence_struct_corpus = result_df[['date','time','tokens']]
sentence_struct_corpus = sentence_struct_corpus[sentence_struct_corpus['tokens'].map(lambda x: (len(x) > 4) and (len(x) < 20))].reset_index()

# create emoji corpus
emoji_corpus = result_df['emojis'].sum()

# create callout corpus
callout_corpus = [x for x in result_df['callouts'].sum() if x != "@all"]

# create website corpus
website_corpus = result_df['websites'].sum()

# choose random sentence struct
import random

def clean_up(element, last_word):
    
    string = element[0]
    if "ahah" in string:
        
        return string
    
    if string == "wo":
        
        return "will"
    
    if string == "n't":
        
        return "not"
    
    if string == "'m":
        
        return "am"
    
    if (string == "'s") and (element[1] != "POS"):
        
        return "is"
    
    if string == "'re":
        
        return "are"
    
    if string == "'ll":
        
        return "will"
    
    if string == "'d":
        
        return "would"
    
    if string == "'ve":
        
        return "have"
    
    if "Emoji" in string:
        
        return re.sub("Emoji",random.choice(emoji_corpus),string)
    
    if "Callout" in string:
        
        return re.sub("Callout",random.choice(callout_corpus),string)
    
    if "Website" in string:
        
        return re.sub("Website",random.choice(website_corpus),string)
    
    return string

def handle_punctuation(sentence, element):
    
    if (element[0] == "``"):
            
        sentence += '"'
            
    elif (element[0] == "''"):

        sentence = sentence[0:(len(sentence) - 1)]
        sentence += '"' + " "

    elif (element[0] in ["?",".","!",",",":",";","...","'","'s"]):

        sentence = sentence.strip()
        sentence += element[0] + " "
        
    elif (element[0] in ["#","$"]):

        sentence += element[0]

    elif (element[0] in ["-"]):
        
        sentence += element[0] + " "
    else:
        
        return {'was_punctuation': False, 'sentence': sentence}
    
    return {'was_punctuation': True, 'sentence': sentence}

def handle_nonwords(element):
    
    if element[0] == "Emoji":

        value = random.choice(emoji_corpus)

    elif element[0] == "Callout":

        value = random.choice(callout_corpus)

    elif element[0] == "Website":

        value = random.choice(website_corpus)
    
    elif element[0] == "i":
        
        value = "i"
        
    else:
        
        return {'was_nonword': False, 'value': ""}
    
    return {'was_nonword': True, 'value': value}

# counter_max: at counter >= counter_max, do not replace the word with a random word
# counter_reset: at counter == counter_reset, reset counter = 0

counter_max = 3
counter_reset = 4

# random everything
def generate_random_message():
    
    structure = random.choice(sentence_struct_corpus['tokens'])
    
    print "STRUCTURE: " 
    print structure
    
    sentence = ""
    first = True
    counter = 0
    
    for element in structure:
        
        print counter
        check_punctuation = handle_punctuation(sentence, element)
        check_nonword = handle_nonwords(element)
        counter = counter + 1
        
        which_pos = ["NN","NNP","NNS","VB","VBN","VBD","VBG","JJ","CD"]
        
        if check_punctuation['was_punctuation']:
            
            sentence = check_punctuation['sentence']
            counter = counter - 1
        
        elif (check_nonword['was_nonword']) and (counter < counter_max):
            
            sentence += check_nonword['value'] + " "
            counter = counter - 1
          
        elif first:
            
            counter = 0
            sentence = element[0] + " "
        # try everything except
        # elif (element[1] not in ["IN","DT","CC", "RB"]) and (len(element[1]) > 1):
        #     sentence += clean_up((random.choice(pos_corpus[element[1]]['word']),element[1])) + " "
        
        # now try only these
        # elif (element[1] in ["NN","NNP","NNS","VBG"]) and (len(element[1]) > 1):
        #     sentence += clean_up((random.choice(pos_corpus[element[1]]['word']),element[1])) + " "
        
        # now try a running counter
        # elif counter < 2:    
        #     sentence += clean_up((random.choice(pos_corpus[element[1]]['word']),element[1])) + " "
            
        # combination only these and counter
        elif (element[1] in which_pos) and (len(element[1]) > 1) and (counter < counter_max):
            sentence += clean_up((random.choice(pos_corpus[element[1]]['word']),element[1]), last_word) + " "
            
        elif (element[1] in which_pos) and (len(element[1]) > 1):
            sentence += clean_up(element, last_word) + " "
            
        elif (len(element[1]) > 1):
            
            counter = counter - 1
            sentence += clean_up(element, last_word) + " "
            
        else:
            
            counter = counter - 1
            sentence = sentence[0:(len(sentence) - 1)]
            sentence += element[0]
            
        if counter >= counter_reset:
            
            counter = 0
    
        first = False
        
        last_word = sentence.split(" ")[-1]
        
    print ""
    print sentence.lower()

while True:

    generate_random_message()
    temp = raw_input("Continue...")

