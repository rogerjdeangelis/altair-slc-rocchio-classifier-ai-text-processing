/* Rocchio classifier corpus — the SAS-native portion of
   altair-slc-rocchio-classifier-ai-text-processing.sas by rogerjdeangelis.
   The two DATA steps below (training documents + labels, and unlabelled
   testing documents) are taken unchanged from the upstream script; WORKX
   is mapped to WORK in autoexec.sas so the run is self-contained. */

options validvarname=v7;
data workx.training;
    input doc & $100. label  $20.;
cards4;
machine learning is great  ai
deep learning is amazing  ai
artificial intelligence is the future  ai
python programming is fun  programming
java programming is robust  programming
c++ programming is fast  programming
;;;;
run;

data workx.testing;
    input doc & $100.;
cards4;
neural networks are powerful
coding in python is enjoyable
java code runs everywhere
deep neural networks learn
;;;;
run;

proc print data=workx.training; run;
proc print data=workx.testing;  run;

/* class balance of the labelled training corpus */
proc freq data=workx.training;
    tables label / nocum;
run;
