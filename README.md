Experiments with models similar to the PBWM model.

http://psych.colorado.edu/~oreilly/papers/OReillyFrank06_pbwm.pdf

Note that learning is done via deep Q-networks as opposed to the PVLV algorithm used in the above paper.  For details on PVLV, see here:

http://ski.clps.brown.edu/papers/OReillyFrankHazyEtAl07.pdf

## dqn 
Main model for store-ignore-recall task.  

* A command to STORE, IGNORE, or RECALL is given along with an integer-valued symbol.  
* The point is to store the symbol when told to store and then recall the symbol when told to recall.
*  A reward is obtained by
    -  1) recalling the correct symbol when told to recall or
    -  2) outputting nothing when told to store or ignore.
* The **use_simplified_task** flag causes RECALL to be given if and only if the previous command was STORE.

## most_trivial_example
Two states (0, 1) and two actions (0, 1) with a reward when the state matches the action.
