import string
from inspection import preprocess
f = '/Users/benno/Documents/ Misc/data/de-en/europarl-v7.de-en.en'

for line in open(f):
	print preprocess(line)
	