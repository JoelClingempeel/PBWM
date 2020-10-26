Experiments with models similar to the PBWM model.

http://psych.colorado.edu/~oreilly/papers/OReillyFrank06_pbwm.pdf

## dqn 
Main model for store-ignore-recall task.  

* A command to STORE, IGNORE, or RECALL is given along with an integer-valued symbol.  
* The point is to store the symbol when told to store and then recall the symbol when told to recall.
*  A reward is obtained by
    -  1) recalling the correct symbol when told to recall or
    -  2) outputing nothing when told to store or ignore.

## baby_dqn
Similar to dqn but where RECALL is given if and only if the previous command is STORE.

## most_trivial_example
Two states (0, 1) and two actions (0, 1) with a reward when the state matches the action.
