#!/bin/bash

# Script to reproduce results
for (( i=0; i<10; i++ ))
do
	scl run data_collection.py \
	--headless \
	--seed $i
done
