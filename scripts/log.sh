for i in $(seq 1 500); 
do  ( ( time python3.4 AdaBoost.py $i) >> ../generated/longTime.log 2>&1 ) &
   if (( $i % 10 == 0 )); then wait; fi
   	 
done
pushover --api-token aqz4SVrrb5a67EwnytQvmfnrYUnifw --user-key uUNPbABuEqPWvR5Y9agZeB59ZiMkqo "Your script is done"
