#!/bin/bash

while read line
do
	echo $line
	input=`echo "$line" | awk -F "\t" '{print $1}'`
	dist1=`echo "$line" | awk -F "\t"  '{print $2}'`
	dist2=`echo "$line" | awk -F "\t"  '{print $3}'`
	dist3=`echo "$line" | awk -F "\t"  '{print $4}'`
	tytul=`echo "$line" | awk -F "\t"  '{print $5}'`

	echo ": $input	$dist1/$dist2/$dist3		$tytul"

	time ./hist-tri.py -i "in/tri/$input" -s "$dist1" -b "$dist2" -r "$dist3" -t "$tytul"
exit
done < in/tri/tri.csv
