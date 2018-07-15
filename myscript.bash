#!/bin/bash

for i in {1..100}; do
	echo "run\nq" | python3 scriptExp.py 11 $i
done
