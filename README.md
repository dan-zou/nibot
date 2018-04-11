# nibot
autochat bot based on madlib randomization
only works on hipchat chat dumps

to prepare

1. download hipchat history from whatever
2. make sure you have the following python packages
  a. numpy
  b. pandas
  c. nltk (and install the relevant corpi and taggers - don't remember which are needed)

to run random-generator.py

1. create/update env.py based on env.py.template
2. python random-generator.py