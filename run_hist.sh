#!/bin/bash

while read line
do
	input=`echo "$line" | awk -F "\t" '{print $1}'`
	dist=`echo "$line" | awk -F "\t"  '{print $2}'`
	tytul=`echo "$line" | awk -F "\t"  '{print $3}'`

	echo "$input	$tytul"

	time ./hist.py -i "in/$input" -d "$dist" -t "$tytul"
#exit
#done < in/biegi.csv
done < in/biegi-pol.csv
