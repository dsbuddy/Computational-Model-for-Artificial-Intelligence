#!/bin/bash

for (( i=1; i<101; ++i )) ; do
	echo "run" | python3 scriptExp.py $(($1+10)) $i
done

# echo "run" | python3 scriptExp.py