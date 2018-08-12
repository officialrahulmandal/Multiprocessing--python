# Fetch labeled data for 4 intents: informational, comparison, purchase, local
# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
'''
Building the queries for each intent:
Informational
Query: what is [common noun]
Data: common nouns are loaded from data/nouns.txt
Note: also filter based on titles (like to accept review in the title)
Comparison
Query: best/top [product categories]
Data: loaded from data/product-categories.txt
Note: Might have to do extra filtering based on title (use new scrapy API)
Purchase:
Query: buy [product]
Data: take top 300 or so product from data/product.csv (the last products are poor quality)
Local:
Query: near me [city]
Data: take random from cities.csv (exclude Puerto rico & alaskan cities)
Total process:
1. Create 200/2500 queries for each intent
2. Get top 500 serps and save to db (parallel)
3. Save the serps (cannot be done in parallel due to pymongo)
4. filter SERPs based on intent specific filter, i.e. no "review" in title of purchase docs (parallel)
500. save final query, url, content, title, intent
For each step, check if already done for given data to avoid processing twice
'''
from multiprocessing import Process, Manager
import random
import pandas as pd
import settings
from pymongo import MongoClient
from IPython import embed
from collections import Counter
import itertools
import re
import multiprocessing
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from functools import partial
import spacy
import re, string
from collections import Counter
from pprint import pprint
from urlparse import urlparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from pymongo.errors import BulkWriteError

# might be removed
import en_core_web_md

from word2vec_uli import main_run
from bson.objectid import ObjectId
from datetime import datetime
from collections import Counter
import joblib
seed = 7199
random.seed(seed)

nlp = spacy.load('en')

DOWNLOAD = True
REMOVE_DUPLICATES = False
URL_COUNT = False
FILTERING = False
FILTERING_ARTICLES = False


#Queries for different intents
info_queries=["what is ","definition " ,"define " ,"example " ,"mean " ,"meaning " , "learn more about " , "information on " ,"explain " ,"learn more "]
comparison_queries = ['best ','what is the best ','top 10 ','which is the best ', 'list of best ', 'the best ', 'list of the best ', 'list of the top ', 'choosing the best ', 'finding the best ']
purchase_queries = ['buy ','purchase ','shop ']
local_queries = ['near me ','in ', 'the area around ','near ','within 1 hour of ','within 5 miles from ', 'what to do in ', 'close to ','where to go in ']


# queries = {}

# # load nouns and create information queries
# queries['information'] = []
# f = open('data/nouns.txt')
# nouns = f.readlines()nss ?
# queries['purchase'] = []
# df = pd.read_csv('data/product.csv')

# # Only take the first half, since quality at the end sucks
# products = df['Name'].tolist()[0:2300]

# indices = random.sample(range(0, len(products)), 250)

# for i in indices:
#   queries['purchase'].append('buy %s' % products[i].strip())


# # load cities.csv and create local queries
# queries['local'] = []
# df = pd.read_csv('data/cities.csv', sep='|')

# # remove cities from puerto rico and alaska
# df = df[df['State short'] != 'PR']
# df = df[df['State short'] != 'AK']

# indices = random.sample(range(0, len(df['State short'].tolist())), 2500)
# cities = df['City'].tolist()
# states = df['State short'].tolist()

# for i in indices:
#   queries['local'].append('near me %s, %s' % (cities[i].strip(), states[i].strip()))

queries = joblib.load("total_queries_Ldata")

for i in queries.keys():
     queries[i] = queries[i][0:20]
     print len(queries[i])

# for i in range(4):
#   print len(queries.values()[i])

# Queries are all done

# Setup mongo db
      
ENV = 'production'

env_dict = settings.ENV[ENV]

if 'MONGO_PASSWORD' in env_dict.keys():
  url = 'mongodb://' + env_dict['MONGO_USER'] + ':' + env_dict['MONGO_PASSWORD'] + '@' + env_dict['MONGO_URL']
else:
  url = 'mongodb://localhost:27017'

print url

# Connect to DB that hold keyword/corpus data
client = MongoClient(url)
db = client[env_dict['MONGO_DB']]



# Filter articles

exclude_list = [
  '<!-- #navbar',
  'Disclaimer',
  'Legal Disclaimer',
  'Fullfillment by Amazon',
  'Access Denied',
  'Ingredients: Ingredients:',
  '{"targetDiv"',
  '12:00 am 12:30 am',
  'View map Updating Map',
  'Tap to Zoom Tap to Zoom',
  'Choose a country',
  'Select Language',
  'address align-top',
  'AZ - Chandler',
  'State Alabama Alaska',
  'Toggle navigation Exploring:',
  'All contents copyright',
  'Change country:',
  'Skip to main content',
  'Disclosures',
  'Set where you live',
  '/~/switchart/',
  'Please enter your birth date',
  ]

intent_class_map = {
  'information': 0,
  'comparison': 1,
  'purchase': 2,
  'local': 3,
  }

replace_map = {
  'PERSON': ' peersoon ',
  'PRODUCT': ' prooduuct ',
  'ORG': ' oorgaaniizaatiioon ',
  'LOC': ' loocaatiioon ',
  'GPE': ' loocaatiioon ',
  'WORK_OF_ART': ' aart '
  }


def do_work(in_queue, out_list):
    while True:
        art = in_queue.get()
        if art == None:
            break
        else:
          result = multipro(art)
          out_list.append(result)

def exact_Match(phrase, complete_articles):
  #phrase = " "+ phrase+" "
  count = 0
  for i in complete_articles.split("###"):
    count+=i.count(phrase)
  if count>0:
    return True
  else:
    return False

# Fetches data from mongo db
# reject article that are too long or short (500 - 1000000 character)
def rawdata_creation(cutoff_rank):
  rejected = []
  bad = []
  final_serps = []
  for intent, i_queries in queries.iteritems():
    # final_serps[intent] = []
    for q in i_queries:
      # embed()
      # if dataset i== 'first_dataset' -> db.serp_dataV2
      # if dataset == 'second_dataset' -> db.serp_dataV2_v2
      serps = db.serp_dataV2.find({'keyword': q, 'article': {'$exists': True}})
      #print q
      for s in serps:
        s['intent'] = intent
        if len(s['article']) > 500 and len(s['article']) < 1000000:
          include = True
          for ex in exclude_list:
            if s['article'].startswith(ex):
              include = False
              bad.append(s)
          if include == True:
            final_serps.append(s)
        else:
          rejected.append(s)
  count_domains={}

  return final_serps

# count top level domains per intent
def count_domains(final_serps):
  domain_count = {}
  for intent in ['information', 'comparison', 'purchase', 'local']:
    domain_count[intent] = []
  for serp in final_serps:
    # print serp['intent']
    # domains=[]
    try:
      domain_count[serp['intent']].append(urlparse(serp["url"]).hostname.replace('www.', ''))
    except:
      domain_count[serp['intent']].append('NoneType')
  for intent in domain_count.keys():
    domain_count[intent] = Counter(domain_count[intent])
  top_domain_perIntent = {}
  for i in domain_count.keys():
    top_domain_perIntent[i] = domain_count[i].most_common()[0:1000]

  return top_domain_perIntent

def raw_string(s):
    if isinstance(s, str):
        s = s.encode('string-escape')
    elif isinstance(s, unicode):
        s = s.encode('unicode-escape')
    return s


# Replace entities with common word
def replace_entities():
  print 'Start replacing entities'
  start_time = datetime.now()
  # Load NLP once in order to avoid loading it many times in the loop
  # nlp = spacy.load('en')

  # Note: Don't use this for tagging...it tags only a fractions
  #nlp = en_core_web_md.load()
  total_ents_tagged = 0

  records = db.serp_dataV2_subset52.find({})
  for serp in records:
    present_keys = serp.keys()
    if "entity_processed_article" in present_keys:
      continue
    else:

      # print serp['url']

      ent_dict = {}
      tmp_article = serp['article']

      remove = [';', '^', '|', '+', ':', '~', '=', '[', ']', '(', ')', '*', '#']
      for r in remove:
        tmp_article = tmp_article.replace(r, ' ')

      doc = nlp(tmp_article)
      ents = []
      for ent in doc.ents:
        #print ent
        if ent.label_ in ['PERSON', 'PRODUCT', 'ORG', 'LOC', 'GPE', 'WORK_OF_ART']:
          try:
            ent_dict[ent.label_].append((ent.text, ent.sent))
          except KeyError:
            ent_dict[ent.label_] = [(ent.text, ent.sent)]

          ents.append((ent.text, ent.label_))

      ents.sort(key = lambda s: len(s[0]), reverse=True)
      total_ents_tagged += len(ents)
      already_processed = {}
      for entity in ents:
        if entity[0] == ' ':
          continue
        try:
          already_processed[entity[0]]
        except KeyError:
          already_processed[entity[0]] = True

        #
        #entity[0].replace(')', '')
        try:
          tmp_article = re.sub(r'[\s.,]' + entity[0] + r'[\s.,]', replace_map[entity[1]], tmp_article)
        except:
          # embed()
          tmp_article = re.sub(r'[\s.,]' + raw_string(entity[0]) + r'[\s.,]', raw_string(replace_map[entity[1]]), tmp_article)
        # Remove quoation marks after entity tagging
        tmp_article = tmp_article.replace('"', '')
        #tmp_article = re.sub(r'\s' + entity[0] + r'\s', replace_map[entity[1]], tmp_article)
      serp['entity_processed_article'] = tmp_article
      # address = intent+ '._id'
      db.serp_dataV2_subset52.update({'_id':serp['_id']}, {'$push': {'entity_processed_article': tmp_article}})

  print 'Total Entities tagged: %i' % total_ents_tagged
  print "Time entity replacement: ",  str(datetime.now() - start_time)



# Load this once for saving time
nlp = spacy.load('en')

# Lemmatize single article
def lemmatize_single_article(article):
  tmp_article = u''

  doc = nlp(article)

  for t in doc:
    if t.is_digit:
      lemma = ' nuumbeer'
    elif t.is_punct:
      lemma = t.lemma_
    elif t.lemma_ == '-PRON-':
      lemma = ' ' + t.string.lower()
    else:
      lemma = ' ' + t.lemma_

    tmp_article += lemma
  tmp_article = re.sub(' +',' ', tmp_article)

  return tmp_article


# Lemmatize all article
def lemmatization():
  print 'Start lemmatization'
  start_time = datetime.now()
  records = db.serp_dataV2_subset52.find({})

  for serp in records:
    present_keys = serp.keys()
    if "lemmatized_article" in present_keys:
      continue
    else:
      # serp['lemmatized_article'] = lemmatize_single_article(serp['query_word_removed_articles'])
      db.serp_dataV2_subset52.update({'_id':serp['_id']}, {'$push': {'lemmatized_article': lemmatize_single_article(serp['query_word_removed_articles'][0]) }})

  print 'Lemmatization done in: %s' %  str(datetime.now() - start_time)

def hasNumbers(inputString):
  return any(char.isdigit() for char in inputString)


# Creates the feature vector for each article in parallel
# Note: Needs the word_id -> keys = word_id.keys() does not work
def multipro2(word_id, art):
  art = " "+ art + " "
  # Initialize all features to 0
  x = [0] * len(word_id)

  for w, idx in word_id.iteritems():
    st = " " + w + " "
    count_i = art.count(st)
    x[idx] = count_i

  return x

def multipro(art):
    # lower case cuz all ngrams are all lower case
    art = art.lower()

    # remove double spaces from article cuz they hinder finding ngrams too
    art = re.sub(' +',' ', art)

    # replace "." and "," cuz they might be in the way when building ngrams
    # Like: "This was bad ever since." might not find "ever_since" because there is a point at the end
    # Note: here we create double spaces again on purpose so words that were seperated by "." or "," do
    # not get converted in ngrams
    art = art.replace('.', ' ')
    art = art.replace(',', ' ')
    art = " "+art+" "

    hits = []
    for i in nonuni_grams:
      i = " " + i + " "
      i1 =  i.replace("_"," ")
      #art = art.replace(i1," "+i+" ")
      art = re.sub(i1,i,art)
      if i in art:
        hits.append(i)

    # Finally, lets remove the double spaces created before
    art = re.sub(' +',' ', art)
    return [art,hits]

def build_string_corpus():
  string_corpus = ''
  records = db.serp_dataV2_subset52.find({})
  # embed()
  for serp in records:
    # print intent

    # for serp in serps:
    string_corpus = string_corpus + serp['lemmatized_article'][0].lower()

  return string_corpus



def building_word_id_dict(keyword_model, min_count):
  a4 = datetime.now()
  word_dict_v2 = {}
  records = db.serp_dataV2_subset52.find({})
  # embed()
  if keyword_model == 1:
    import spacy
    nlp = spacy.load('en')
  if keyword_model == 2:
    import en_core_web_md
    nlp = en_core_web_md.load()

  if keyword_model == 1 or keyword_model == 2:
    a8 = datetime.now()
    doc = nlp(complete_articles)


    for t in doc:
      if t.is_digit == False and t.is_punct == False and t.is_space == False and t.like_url == False and t.like_email == False:
        # if t.text == 'locationy':
        # a = a + b
        try:
          c[t.lemma_] += 1
          word_dict_v2[t.lemma_]
        except KeyError:
          word_dict_v2[t.lemma_] = word_index
          word_index += 1
    b8 = datetime.now()
    print "spacy run time: ", str(b8-a8)

    # Create word - id dictionary
    idx = 0
    word_id = dict()
    for w, count in c.iteritems():
        if count > min_count:
          try:
            word_id[w]
          except KeyError:
            word_id[w] = idx
            idx += 1

  # condition for word2vec feature model -- variable names needto be changed
  else:
    # Build ngrams
    a5 = datetime.now()
    a7 = datetime.now()

    # build pure string corpus so ngrams can be extracted
    string_corpus = build_string_corpus()

    word2vec = main_run(string_corpus)
    b7 = datetime.now()
    print "word2vec creation run time : ", str(b7-a7)

    # Determine ngram distribution of word_id (cleaned by min count)
    unigrams = []
    bigrams = []
    trigrams = []
    fourgrams = []
    fivegrams = []
    ngrams_distribution = {}
    ngrams_dict = {}
    word2vec_keys = word2vec.keys()
    print(len(word2vec_keys))

    for i in word2vec_keys:
      ngrams = i.split("_")
      if len(ngrams) == 1:
        unigrams.append(i)
      elif len(ngrams) == 2:
        bigrams.append(i)
      elif len(ngrams) == 3:
        trigrams.append(i)
      elif len(ngrams) ==4:
        fourgrams.append(i)
      elif len(ngrams) ==5:
        fivegrams.append(i)
      else:
        continue

    ngrams_dict['unigrams'] = unigrams
    ngrams_dict['bigrams'] = bigrams
    ngrams_dict['trigrams'] = trigrams
    ngrams_dict['fourgrams'] = fourgrams
    ngrams_dict['fivegrams'] = fivegrams

    ngrams_distribution["unigram count"] = len(unigrams)
    ngrams_distribution["bigram count"] = len(bigrams)
    ngrams_distribution["trigram count"] = len(trigrams)
    ngrams_distribution["fourgram count"] = len(fourgrams)
    ngrams_distribution["fivegram count"] = len(fivegrams)
    print 'Ngrams distribution of pure word2vec ngrams extraction before min_count'
    print "unigram count: " + str(len(unigrams))
    print "bigram count: " + str(len(bigrams))
    print "trigram count: " + str(len(trigrams))
    print "fourgram count: " + str(len(fourgrams))
    print "fivegram count: " + str(len(fivegrams))
    print


    print 'Total phrase from word2vec ngram creation: %i' % len(word2vec)


    # Create final word/ngram dictionary based on min_count
    # TODO: different min_count for unigrams and multigrams
    word_id = dict()
    idx = 0
    stop_words = [w.strip() for w in open('modified_stoplist.txt').readlines()]
    for w, count in word2vec.iteritems():
      gramminess = len(w.split("_"))
      if gramminess < 6:
        # filter out stopwords
        if gramminess == 1 and w in stop_words:
          continue

        # filter out numbers not recognized by spacy
        if hasNumbers(w):
          continue

        if count >= min_count[str(gramminess)]:
          #if w in set(prelim_keys):
          #if exact_Match(w,complete_articles) == True:
          z = w
          word_id[z] = idx
          idx += 1

    print len(word_id)


    # Determine ngram distribution of word_id (cleaned by min count)
    unigrams = []
    bigrams = []
    trigrams = []
    fourgrams = []
    fivegrams = []
    ngrams_distribution = {}
    ngrams_dict = {}
    word2vec_keys = word_id.keys()
    print(len(word2vec_keys))

    for i in word2vec_keys:
      ngrams = i.split("_")
      if len(ngrams) == 1:
        unigrams.append(i)
      elif len(ngrams) == 2:
        bigrams.append(i)
      elif len(ngrams) == 3:
        trigrams.append(i)
      elif len(ngrams) ==4:
        fourgrams.append(i)
      elif len(ngrams) ==5:
        fivegrams.append(i)
      else:
        print''

    ngrams_dict['unigrams'] = unigrams
    ngrams_dict['bigrams'] = bigrams
    ngrams_dict['trigrams'] = trigrams
    ngrams_dict['fourgrams'] = fourgrams
    ngrams_dict['fivegrams'] = fivegrams

    ngrams_distribution["unigram count"] = len(unigrams)
    ngrams_distribution["bigram count"] = len(bigrams)
    ngrams_distribution["trigram count"] = len(trigrams)
    ngrams_distribution["fourgram count"] = len(fourgrams)
    ngrams_distribution["fivegram count"] = len(fivegrams)
    print 'Ngrams distribution of word_id'
    print "unigram count: " + str(len(unigrams))
    print "bigram count: " + str(len(bigrams))
    print "trigram count: " + str(len(trigrams))
    print "fourgram count: " + str(len(fourgrams))
    print "fivegram count: " + str(len(fivegrams))

    countt =0
    global nonuni_grams
    nonuni_grams = fivegrams + fourgrams + trigrams + bigrams

    pool = multiprocessing.Pool(1)
    all_hits = []

    # for intent, serps in final_serps.iteritems():

    a6 = datetime.now()
    total_serps=[]
    p=0
    for serp in records:
      p+=1
      total_serps.append(serp['lemmatized_article'][0])
    countt+=len(total_serps)

    #######
    num_workers = 5
    manager = Manager()
    output = manager.list()
    work = manager.Queue(num_workers)
    pool_lst = []
    for i in range(num_workers):
      p = Process(target=do_work, args=(work, output))
      pool_lst.append(p)
      p.start()
    articles = itertools.chain(articles, (None,)*num_workers)
    for i in total_serps:
      work.put(i)

    for k in pool_lst:
      k.join()
    #######
    #func = partial(multipro, nonuni_grams)
    #output = pool.map(func, total_serps)
    # embed()

    total_serps = []

    # TODO: there must be a better way to do this with queues
    for out in output:
      total_serps.append(out[0])
      all_hits.extend(out[1])

    records = db.serp_dataV2_subset52.find({})
    a=0
    p=0

    # embed()
    for serp,t_serp in zip(records,total_serps):
      db.serp_dataV2_subset52.update({'_id':serp['_id']}, {'$push': {'articles_with_ngrams': t_serp }}) 

    b6 = datetime.now()
    # print "skipgram articles_with_ngrams for a intent run time : ", intent, " : ", str(b6-a6)

    # TODO: make into own method
    idx = 0
    all_hits = [w.strip() for w in list(set(all_hits))]
    new_word_id = {}
    for w, old_id in word_id.iteritems():
      gramminess = len(w.split('_'))
      if (gramminess == 1):
        new_word_id[w] = idx
        idx += 1
      elif w in all_hits:
        #print w
        new_word_id[w] = idx
        idx += 1
      else:
        continue

    joblib.dump(word_id, "complete_wordID")
    joblib.dump(new_word_id, "filtered_wordID")
    print 'Old word_id size: %i' % len(word_id)
    print 'New word_id size: %i' % len(new_word_id)


    # only include features that got hits
    b5 = datetime.now()
    print "skipgram run time : ", str(b5-a5)


  print "total articles count: ", countt
  #print "Length of word_id dict: ", len(word_id)
  b4 = datetime.now()
  print "building_word_id_dict run time: ", str(b4-a4)
  return ngrams_distribution, ngrams_dict


def text_to_feature_vector(word_id, article):
  return [i for i in range(10)]
  x = [0] * len(word_id)
  remove = [';', '^', '|', '+', ':', '~', '=', '[', ']', '(', ')', '*', '#']
  for r in remove:
    article = article.replace(r, ' ')

  keys = word_id.keys()
  for i in range( len(word_id)):
    st = '(?:\s|^)' + keys[i] + '(?:\s|$)'
    count_i =len(re.findall(st, article))
    x[i] += count_i
  return x

def count_articles(final_serps):
  article_count_preintent = {}
  for intent in ['information', 'comparison', 'purchase', 'local']:
    article_count_preintent[intent] = 0
  for serp in final_serps:
    article_count_preintent[serp['intent']] = article_count_preintent[serp['intent']] + 1

  return article_count_preintent


# removes the query words from text
# TODO: might extend this by excluding words that have close cosine distance to query word!
# TODO: clean up
def replace_query_words():
  print 'Start replace_query_words'

  start_time = datetime.now()

  intent_query_map = {
    'information': ["what is ","definition " ,"define " ,"example " ,"mean " ,"meaning " , "learn more about " , "information on " ,"explain " ,"learn more "],
    'comparison': ['best ','what is the best ','top 10 ','which is the best ', 'list of best ', 'the best ', 'list of the best ', 'list of the top ', 'choosing the best ', 'finding the best '],
    'local': ['buy ','purchase ','shop '],
    'purchase': ['near me ','in ', 'the area around ','near ','within 1 hour of ','within 5 miles from ', 'what to do in ', 'close to ','where to go in ']
  }

  records = db.serp_dataV2_subset52.find({})
  for serp in records:
    present_keys = serp.keys()
    if "query_word_removed_articles" in present_keys:
      continue
    else:
      tmp_article = serp['entity_processed_article'][0].lower()
      pre_keywords = intent_query_map[serp['intent']]
      for i in pre_keywords:
        if i in serp['keyword']:
          query = serp['keyword'].replace(i + ' ', '').lower()

      # remove unwanted signs from query
      remove = [';', '^', '|', '+', ':', '~', '=', '[', ']', '(', ')', '*', '#', '.', ',', '&']
      for r in remove:
        query = query.replace(r, ' ')


      # remove double space
      query = re.sub(' +',' ', query)
      words_in_query = query.split(' ')
      for w in words_in_query:
        # skip short words like state abbreviations in query (IN, NE, MA)
        if (len(w) <= 2):
          continue
        w = ' ' + w + ' '
        tmp_article = tmp_article.replace(w, ' thetopic ')



      # Replace multiple following instance of "thetopic" into one
      # Some topic/queries are within quotation marks
      tmp_article = tmp_article.replace("'", '')
      tmp_article = tmp_article.replace('"', '')
      tmp_article = re.sub(' +',' ', tmp_article)

      #print tmp_article

      # Remove double spaces, just to be sure
      tmp_article = re.sub(' +',' ', tmp_article)
      tmp_article = tmp_article
      for i in reversed(range(1,5)):
        to_be_replaced = ' thetopic' * i
        tmp_article = tmp_article.replace(to_be_replaced, ' thetopic ')


      # remove double space
      tmp_article = re.sub(' +',' ', tmp_article)

      if 'thetopictthet' in tmp_article:
        embed()
        a = a

      serp['query_word_removed_articles'] = tmp_article
      db.serp_dataV2_subset52.update({'_id':serp['_id']}, {'$push': {'query_word_removed_articles': tmp_article}})

  print 'Finished in: %s' % (datetime.now() - start_time)


def count_words(final_serps):
  print 'Counting total words'
  word_count = 0
  for serp in final_serps:
    # print intent
    word_count += len(serp['article'].split(' '))

  return word_count

# Create training data
def training_data_creation(cutoff_rank, keyword_model, min_count):
  a9= datetime.now()

  final_serps = rawdata_creation(cutoff_rank)

  top_domain_perIntent = count_domains(final_serps)
  article_count_preintent = count_articles(final_serps)
  total_word_count = count_words(final_serps)
  # embed()

  # TODO: extract into separate methods
  p1=0
  map_dict_url={}
  map_dict_art={}

  # for intent, serps in final_serps.iteritems():

  a10 = datetime.now()
  total_serps=[]

  for serp in final_serps:
      map_dict_url[p1] = serp['url']
      map_dict_art[p1] = serp['article']
      p1+=1

  #making entries unique
  #final_serps=list(np.unique(np.array(final_serps)))
  for i in range(0,len(final_serps)):
    final_serps[i]["_id"] = i

  print "begin bulk write"
  #db.serp_dataV2_subset52.insert_many(final_serps)
  print "bulk write done"


  replace_entities()
  replace_query_words()
  lemmatization()
  # embed()
  ngrams_distribution, ngrams_dict = building_word_id_dict(keyword_model, min_count)

  # Create feature vectors

  k=0
  vec_sum=0

  X = []
  Y = []
  # Needed for tfidf and later for extraction of
  vectorizer = 0

  if keyword_model == 1  or keyword_model == 2:
    for intent, serps in final_serps.iteritems():
      for serp in serps:
        x = text_to_feature_vector(word_id, serp['article'])
        if sum(x) ==0:
          vec_sum+=1
        X.append(x)
        Y.append(intent_class_map[intent])
  elif keyword_model == 'count':
    # feature vector creation for ngram count vectors
    p=0

    # Big bug here!!!!
    #keys  = word_id.keys()

    pool2 = multiprocessing.Pool(4)

    for intent, serps in final_serps.iteritems():

      a10 = datetime.now()
      total_serps=[]

      for serp in serps:
          total_serps.append(serp['articles_with_ngrams'])
          Y.append(intent_class_map[intent])

      func2 = partial(multipro2, word_id)
      out1 = pool2.map(func2, total_serps)

      for x in out1:
        x.append(p)
        if sum(x) ==0:
          vec_sum+=1
        X.append(x)
        p+=1
      b10 = datetime.now()
      print "skip gram intent vector creation run time: ",  str(b10-a10)

  elif keyword_model == 'tfidf':
    # ngram tdidf feature vector creattion
    records = db.serp_dataV2_subset52.find({})
    corpus = []
    for serp in records:
      # embed()
      corpus.append(serp['articles_with_ngrams'][0])
      # Create Y
      Y.append(intent_class_map[serp['intent']])


    word_id = joblib.load( "filtered_wordID")
    stop_words = [w.strip() for w in open('modified_stoplist.txt').readlines()]
    vectorizer = TfidfVectorizer(stop_words=stop_words, vocabulary=word_id)
    X = vectorizer.fit_transform(corpus)




  #print "No. of vectors or No. of articles: ", len(X)
  #print "No. of complete zero vectors: ", vec_sum
  b9 = datetime.now()
  print "training_data_creation run time: ", str(b9-a9)
  return X, Y, final_serps, word_id, map_dict_url, map_dict_art, top_domain_perIntent, ngrams_distribution, article_count_preintent, vectorizer
