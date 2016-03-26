#!/usr/bin/env python
# -*- coding: utf-8 -*-

### config ###

dpi=80

### body ###

'''
instalation

pip freeze | egrep "numpy|matplotlib|pandas|prettyplotlib"
matplotlib==1.4.3
numpy==1.9.2
pandas==0.16.0

sudo apt-get install libfreetype6-dev libxft-dev
pip install numpy pandas matplotlib prettyplotlib

'''

import sys
import getopt

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FuncFormatter, MultipleLocator
import pandas as pd
import numpy as np
import datetime
#import prettyplotlib as ppl
#import brewer2mpl
import os, errno
import seaborn as sns
from string import Template	# html templates
import codecs	# unicode write-save files

sns.set(style="ticks")


input_csv = ''
global_tytul = ''
TRESC = ""
MENU = ""


try:
    myopts, args = getopt.getopt(sys.argv[1:],"i:t:")
except getopt.GetoptError as e:
    print (str(e))
    usage = "Usage: %s\n\t-i input csv file\n\t-t tytuł" % sys.argv[0]
    print usage
    sys.exit(2)

for o, a in myopts:
    if o == '-i':
        input_csv=a.decode('UTF-8')
    elif o == '-t':
        global_tytul=a.decode('UTF-8')
    else:
	print usage
	sys.exit(2)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def getSec(s):
    if(not pd.isnull(s)):  
      if len(s) == 5:
	s = "00:" + s
      l = str(s).split(':')
      #print l[0], l[1],l[2]
      return int(l[0]) * 3600 + int(l[1]) * 60 + float(l[2])

def getHMS(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def time_ticks(x, pos):
    d = datetime.timedelta(seconds=x)
    return str(d)

################################################
# create output dir
outputdir_rel = "out/" + os.path.splitext(os.path.basename(input_csv))[0] + "/"
#outputdir = "/mnt/nfs/www/statystykibiegowe/"
outputdir = "OUT_HTML/"
mkdir_p(outputdir + outputdir_rel)

print outputdir_rel

################################################
wej = pd.read_csv(input_csv, sep=",", encoding='utf-8') #, index_col=[0])

t_min = np.min(wej.czas_netto.apply(getSec))
t_max = np.max(wej.czas_netto.apply(getSec))


if t_max > 8*3600:	# bardzo długie biegi
  print "Bardzo długi bieg"
  dT = 1200

elif t_max > 6*3600:
  print "Długi bieg, odcinamy czasy >6h"
  t_max = 6*3600 # odcinamy czasy > 6h
  dT = 300

else:
  print "Normalny bieg"
  dT = 300


bin_min = int(t_min/dT)*dT
bin_max = int(t_max/dT+1)*dT
bins = np.arange(bin_min, bin_max + dT, dT)

print bins


uczestnikow = len(wej.index)
wej['czas_netto_s'] = wej.czas_netto.apply(getSec)
#statsDf = pd.DataFrame(wej.czas_netto_s.describe()[1:].apply(getHMS))
print "Uczestników:", uczestnikow

if 'plec' not in wej:
  wej['plec'] = 'wszyscy'
  plcie = { "wszyscy": ['wszyscy']}
else:
  plcie = { "wszyscy": ['K','M'], "kobiety": ['K'], u"mężczyźni": ['M'] }
  wej.plec.replace('F', 'K', inplace=True)

### normalne histogramy ###


TRESC += "<a name='histogramy'><h2>Histogramy</h2></a>"
MENU += "<a href='#histogramy'>Histogramy</a></br>"

for plecOpis, plec in plcie.iteritems():
    df = wej[wej.plec.isin(plec)]
    opis="+".join(plec)

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(bottom=0.18)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.xaxis.set_major_formatter(FuncFormatter(time_ticks))
    ax.xaxis.set_major_locator(MultipleLocator(dT))
    plt.title("%s (%s)" % (global_tytul, opis))
    plt.xlabel("Czas netto (s)")
    plt.ylabel(u"zawodników")
    plt.xticks(rotation='vertical')
    ax.fmt_xdata = DateFormatter('%H:%M')
    sns.distplot(df.czas_netto_s, rug=True, bins=bins);

    #ppl.hist(np.asarray(df.czas_netto_s), grid='y', color='orange', bins=bins)
    outFileName = "hist-%s.png" % (opis)
    plt.savefig(outputdir + outputdir_rel + outFileName, dpi=dpi)

    TRESC += u"<h3>%s</h3>\n" % (plecOpis)
    TRESC += u"<p><img src='%s' alt='%s' /></p>\n" % (outFileName, global_tytul)
    plt.clf()


TRESC += u"<a name='rybkowe'><h2>Wykresy rybkowe</h2></a>\n"
MENU += "<a href='#rybkowe'>Wykresy rybkowe</a><br />"

def rysuj_violin_plot(groupby, tytul, wysokosc=5, warunek=50):
   print "***", groupby
   if groupby in wej:
      zgrupowane = wej.groupby(groupby)
      zgrupowane = zgrupowane.filter(lambda x: len(x) >= warunek)	# tylko takie z > n przypadków
      ileGrup = len(zgrupowane.groupby(groupby))	# powtórne grupowanie, bo się rozgrupowało
      #ileGrup = len(zgrupowane)
      print "ileGrup", ileGrup
      if(ileGrup > 1 and ileGrup <= 30):
		  
	  print "Kolumna %s istnieje, <=30 grup, robimy violinploty!" % (groupby)
	  global TRESC, MENU, global_tytul
	  
	  print "Liczba grup:", ileGrup
	  TRESC += u"<a name='%s'><h3>%s</h3></a>\n" % (groupby, tytul)

	  sns.set_style("whitegrid")
	  fig, ax = plt.subplots(figsize=(10, wysokosc))
	  plt.subplots_adjust(bottom=0.16)

	  ax.xaxis.set_major_formatter(FuncFormatter(time_ticks))
	  ax.xaxis.set_major_locator(MultipleLocator(600))

	  plt.title(global_tytul + " :: " + tytul)
	  plt.xticks(rotation='vertical')
	  ax.fmt_xdata = DateFormatter('%H:%M')

	  sns.violinplot(data=zgrupowane, x="czas_netto_s", y=groupby, palette="Set1", orient='h', inner="quartile", bw=0.1)
	  outplik = "violinplot-%s.png" % groupby
	  plt.savefig(outputdir + outputdir_rel + outplik, dpi=dpi)

	  TRESC += "<p><img src='%s' alt='%s' /></p>" % (outplik, tytul)
	  MENU += u"∙ <a href='#%s'>%s</a><br />" % (groupby, tytul)


	  ##### SWARM 
	  if len(wej) < 1500:
	      plt.clf()
	      sns.set_style("whitegrid")
	      fig, ax = plt.subplots(figsize=(10, wysokosc))
	      plt.subplots_adjust(bottom=0.16)

	      ax.xaxis.set_major_formatter(FuncFormatter(time_ticks))
	      ax.xaxis.set_major_locator(MultipleLocator(600))

	      plt.title(global_tytul + " :: " + tytul)
	      plt.xticks(rotation='vertical')
	      ax.fmt_xdata = DateFormatter('%H:%M')


	      sns.swarmplot(x="czas_netto_s", y=groupby, hue=groupby, data=zgrupowane, palette="Set1")
	      outplik = "swarmplot-%s.png" % (groupby)
	      plt.savefig(outputdir + outputdir_rel + outplik, dpi=dpi)
		      
	      TRESC += "<p><img src='%s' alt='%s' /></p>" % (outplik, tytul)

	  ### STATYSTYKI HTML ####

	  #TRESC += (zgrupowane.describe().to_html().replace(u"czas_netto_s", u"wszyscy")) #.encode('utf-8').strip()
	  #columnsToKeep = [1,2,3,4,5,6,7,9,10,11,12,13,14,15]
	  columnsToKeep = [1,2,3,4,5,6,7]
	  #TRESC += pd.DataFrame(zgrupowane.groupby(groupby).czas_netto_s.describe()[columnsToKeep].apply(getHMS)).rename(columns={0: 'czas'}).to_html()
	  statsy = pd.DataFrame(zgrupowane.groupby(groupby).czas_netto_s.describe().apply(getHMS)).rename(columns={0: 'czas'}).unstack()[columnsToKeep] #.to_html()
	  statsy['count'] =  zgrupowane.groupby(groupby).count().ix[:, 0]
	  TRESC += statsy.to_html()
	  #exit(1)
	  


########## MIĘCHO

rysuj_violin_plot(groupby='plec', tytul=u"wg płci", warunek=1)
rysuj_violin_plot(groupby='kat', tytul="wg kategorii", wysokosc=10, warunek=5)
rysuj_violin_plot(groupby='kraj', tytul=u"wg krajów", wysokosc=10, warunek=10)


if(uczestnikow > 7500):
    rysuj_violin_plot(groupby='imie', tytul="wg imion", warunek=100, wysokosc=10)
    rysuj_violin_plot(groupby='nazwisko', tytul=u"wg nazwisk", warunek=20, wysokosc=10)
    rysuj_violin_plot(groupby='team', tytul=u"wg teamów", warunek=10, wysokosc=10)
    rysuj_violin_plot(groupby='miejscowosc', tytul=u"wg miejscowości", warunek=50, wysokosc=10)


#### polishing html
  
#TRESC = TRESC.encode('utf-8').strip()


#### writing html

f = codecs.open('template.tpl', encoding='utf-8')
src = Template( f.read() )

#src = Template('$TRESC $DATA')


#d = { 'TYTUL':global_tytul, 'UCZESTNIKOW':uczestnikow, 'DATA': datetime.datetime.now(), 'TRESC':TRESC, 'RELATIVE':"../"}
d = { 'TYTUL':global_tytul, 'UCZESTNIKOW':uczestnikow, 'DATA': datetime.datetime.now(), 'TRESC':TRESC, 'MENU':MENU, 'RELATIVE':"../../"}

f = codecs.open( outputdir + outputdir_rel + "index.html", 'w', encoding='utf-8' )
out = src.substitute(d) #.encode('utf-8').strip()
f.write(out)
f.close()

