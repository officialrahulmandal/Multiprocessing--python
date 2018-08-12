import threading
import Queue
import re
import time
start_time = time.time()

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
    for i in ordered_ngrams:
      i = " " + i + " "
      i1 =  i.replace("_"," ")
      #art = art.replace(i1," "+i+" ")
      art = re.sub(i1,i,art)
      if i in art:
        hits.append(i)

    # Finally, lets remove the double spaces created before
    art = re.sub(' +',' ', art)
    return [art, hits]

global ordered_ngrams
nonuni_grams = ["i_am","go_out"]
nonuni_grams = nonuni_grams*3000
#would make this nonunigrams list as global
ordered_ngrams=nonuni_grams

articles = ["i am going back to city","let's go out somewhere"]
articles = articles*200000

def do_work(in_queue, out_queue):
    while True:
        item = in_queue.get()
        if item == None:
            break
        result = multipro(item)
        out_queue.put(result)
        in_queue.task_done()

work = Queue.Queue()
results = Queue.Queue()
num_threads=32
threads=[]
for i in xrange(num_threads):
    t = threading.Thread(target=do_work, args=(work, results))
    t.daemon = True
    t.start()
    threads.append(t)
    
for i in articles:
    work.put(i)
work.join()

#we are getting results which we would append in list later on final code
x = []
for i in articles:
    x.append(results.get())
print("--- %s seconds ---" % (time.time() - start_time))
print "task_done"

