#!/bin/bash

while read line
do
	input=`echo "$line" | awk -F "\t" '{print $1}'`
	tytul=`echo "$line" | awk -F "\t"  '{print $2}'`

	echo "$input	$tytul"

	time ./hist.py -i "in/$input" -t "$tytul"
#exit
done < in/biegi.cfg
