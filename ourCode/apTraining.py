dims = 5
import numpy

# posExamples
# negExamples


data = [[[0.1,0.2,0.3,0.1,0],True],[[-0.1,0.2,-0.3,0.1,0],False],[[0.1,0.2,0.3,0.1,0],False],[[0.1,0.2,0.3,0.1,0],True],[[0.1,0.2,0.3,0.1,0],True]]


iters = 10

weights  = numpy.random.rand(dims)
aweights = numpy.random.rand(dims)
print 'weights:', weights
print 'cached weights:', aweights




coeff = 1.0 / (len(data)*iters)

remaining = len(data) * iters

while iters > 0:
    print 'nr of iterations to go:', iters
    numUpdt = 0
    for vector, label in data:
        prediction = numpy.inner(weights,vector)>0
        if prediction < label:
        # predicted positive, but label is negative
            weights  = weights  + [value * (1-coeff) for value in vector]
            aweights = aweights + [value * (1-coeff*remaining*coeff) for value in vector]
            numUpdt += 1
        if prediction > label:
        # predicted negative, but label is positive
            weights  = weights  - [value * (1-coeff) for value in vector]
            aweights = aweights - [value * (1-coeff*remaining*coeff) for value in vector]
            numUpdt += 1
        remaining -= 1
    print 'udated',numUpdt,'times.'
    print 'weights:', weights
    print 'cached weights:', aweights
    iters -= 1

