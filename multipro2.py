import multiprocessing
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import time
from multiprocessing import Process, Manager
global ordered_ngrams
nonuni_grams = ["i_am","go_out","new_york"]
nonuni_grams = nonuni_grams*100
#would make this nonunigrams list as global
ordered_ngrams=nonuni_grams	
start_time = time.time()

articles = ["i am going back to city","let's go out somewhere", "i am not going to new york"]
from functools import partial
import re
def multipro(art):
    # lower case cuz all ngrams are all lower case
    return "abcdef"
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
    for i in ordered_ngrams:
      i = " " + i + " "
      i1 =  i.replace("_"," ")
      #art = art.replace(i1," "+i+" ")
      art = re.sub(i1,i,art)
      if i in art:
        hits.append(i)

    # Finally, lets remove the double spaces created before
    art = re.sub(' +',' ', art)
    return [art,hits]




def do_work(in_queue, out_list):
    while True:
        art = in_queue.get()
        if art == None:
            break
        else:
          result = multipro(art)
          out_list.append(result)
          # return result



import itertools

num_workers = 3
manager = Manager()
results = manager.list()
work = manager.Queue(num_workers)
pool_lst = []
for i in range(num_workers):
    p = Process(target=do_work, args=(work, results))
    pool_lst.append(p)
    p.start()

articles = articles*5000000

articles = itertools.chain(articles, (None,)*num_workers)

for i in articles:
    work.put(i)



for k in pool_lst:
    k.join()

#print results
print("--- %s seconds ---" % (time.time() - start_time))
print "task_done"
