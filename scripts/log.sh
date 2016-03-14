rm ../generated/longTime.log
for i in $(seq 1 10 500); 
do  ( (  python3.4 AdaBoost.py $i) >> ../generated/longTime.log 2>&1 ) &
   if (( $i % 100 == 0 )); then wait; fi
   	 
done
pushover --api-token aqz4SVrrb5a67EwnytQvmfnrYUnifw --user-key uUNPbABuEqPWvR5Y9agZeB59ZiMkqo "Your script is done"
