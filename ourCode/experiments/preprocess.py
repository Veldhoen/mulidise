import string

def preprocess(s):
	return s.lower().translate(string.maketrans("",""), string.punctuation)

f = '/Users/benno/Documents/ Misc/data/de-en/europarl-v7.de-en.en'

for line in open(f):
	print preprocess(line)
	