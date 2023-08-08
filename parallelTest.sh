#!/bin/bash

NETWORK=1081
PARALLEL=1

for i in {2000..2155}
do
    echo "$i"
    cat test_template.py | sed "s/SOMENUMBER/$i/g" > test_$i.py
    sed -i "s/SOMENET/$NETWORK/g" test_$i.py
    { python test_$i.py; rm test_$i.py; } &
#    { sleep 15; rm test_$i.py; } &
    if [[ $(expr $i % $PARALLEL) == "0" ]]; then
        wait
    fi
done
