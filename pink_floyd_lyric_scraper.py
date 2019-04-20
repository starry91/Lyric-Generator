from bs4 import BeautifulSoup as bs
import urllib
import re
import os
import pandas as pd


URL = 'https://www.allthelyrics.com'
f = open('the_beatles_lyrics.txt', 'a+')

soup = bs(urllib.request.urlopen(urllib.request.Request(URL+'/lyrics/the_beatles')))
links = soup.find_all('a')
songs_list = []
useful_links = [URL+link['href'] for link in links if '/lyrics/the_beatles' in link['href']]

for i, link in enumerate(useful_links):
    print('{}-->{}'.format(i, link))
    try:
        soup = bs(urllib.request.urlopen(urllib.request.Request(link)))
        text = soup.find_all('p')
        song =''
        for text in text[:-4]:
            txt = text.getText()
            f.write(txt)
            song += txt
#             print(txt)
        f.write('\n\n\n')
        songs_list.append([link,song])
    except:
        print('Could not scrape for %dth link %s' %(i, link))
df = pd.DataFrame(songs_list, columns =['url', 'text'])
df.to_csv("pink_floyd_lyrics.csv", sep='\t')
f.close()
