import numpy
import numpy.random

data = numpy.random.randn(1000,2)
labels = numpy.zeros(1000)

data[:250] += numpy.array([-5,5])
data[250:500] += numpy.array([5,5])
data[500:750] += numpy.array([5,-5])
data[750:] += numpy.array([-5,-5])

labels[:250] = 0
labels[250:500] = 1
labels[500:750] = 2
labels[750:] = 3


numpy.save("data",data)
numpy.save("labels",labels)
