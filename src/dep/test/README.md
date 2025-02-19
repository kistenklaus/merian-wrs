The directory contains the testing suit, 
before running any tests the testing suit should validate 
it's correctness by testing against the references.

All assertions implemented in the is_* files check for 
invariants and returns a error, which evaluates to true 
if a error exists and error.message() gives a pretty error message 
to log to the console.
