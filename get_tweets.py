import datetime, time
import re
from test import *
import csv
from twitter_tokenizer import *

TIMEOUT = 0

tweets = {}
tweets_features = {}
count = 0

def get_tweets(tw_ids,timeout=TIMEOUT):  
    global count 
    global tweets

    icount = 0     
    for i in range(len(tw_ids)/100):    
        if icount % 20 == 0:
            print i, "'th batch being processed" 
        tws = twitter_api.statuses_lookup(tw_ids[i*100:min((i+1)*100, len(tw_ids))])
        for t in tws:
            tweets_features[t.id] = [int(t.user.id_str), str(t.created_at), u' '.join(tokenizeRawTweetText(t.text)).encode('utf-8')]
        time.sleep(timeout)
    print count

if __name__ == "__main__":
    labelmap = {"positive":1, "negative":-1, "neutral":2, "irrelevant":2}
    labelcount = {}
    with open('/Users/PauKung/Downloads/sanders-twitter-0.2/corpus.csv','rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting = csv.QUOTE_MINIMAL)
        for row in reader:
            tweets[int(row[2])]=[row[0], row[1]]
            label = labelmap[row[1]]
            try:
                labelcount[label] += 1
            except Exception:
                labelcount[label] = 1
    print labelcount
    
    print "finished loading pickled data"
    get_tweets([tid for tid in tweets])
    with open('/Users/PauKung/Downloads/sanders-twitter-0.2/features.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for tw in tweets_features:
            label = labelmap[tweets[tw][1]]
            try:
                labelcount[label] += 1
            except Exception:
                labelcount[label] = 1
            out = [label]
            out.extend(tweets_features[tw])
            writer.writerow(out)
    print labelcount
    
