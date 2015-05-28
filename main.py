import theano as T
import numpy as np
from scipy.spatial.distance import cosine
from scipy.linalg import norm
from blocks.bricks.cost import SquaredError
from blocks import initialization
from blocks.bricks import Identity, Tanh, Linear
from blocks.model import Model
from blocks.bricks.recurrent import SimpleRecurrent, BaseRecurrent, recurrent, LSTM, GatedRecurrent
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.bricks import WEIGHT
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.algorithms import GradientDescent, Scale, Momentum, RMSProp
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.roles import INPUT, DROPOUT
from blocks.extensions.plot import Plot
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.graph import apply_dropout
from fuel.transformers import Padding, Mapping
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.datasets import IterableDataset, IndexableDataset
from collections import OrderedDict
from random import shuffle

def swap01(data):
    features, targets = data
    return (features.swapaxes(0,1), targets)

class WordBank:

    def __init__(self, words, vector_size=10):
        """ given list of words, generate random hashes and store them """

        self.word_vectors = {}
        for word in words:
            self.word_vectors[word] =  np.random.uniform(size=vector_size)

    def get_closest_words(self, search_vector, num_words=5):
        distances = {}
        for word, vector in self.word_vectors.iteritems():
            distances[word] = abs(cosine(self.word_vectors[word],search_vector))

        distances = OrderedDict(sorted(distances.iteritems(), key=lambda x: x[1]))

        return distances.keys()[:num_words]

    def insert_word(self, word):
        self.word_vectors[word] = np.random.uniform(size=vector_size)

    def get_vector(self, word):
        try:
            return self.word_vectors[word]
        except:
            return None

    def set_vector(self, word, vector):
        self.word_vectors[word] = vector
        return True

    def set_nearest_vector(self, nearby_vector, replace_vector):
        word = self.get_closest_words(nearby_vector, 1)[0]
        self.word_vectors[word] = replace_vector
        return True


    def convert_to_vectors(self, sentence):
        """ expects single string like 'the quick brown fox jumped over the yellow house' 
            returns a list of vectors
        """
        return [self.word_vectors[word] for word in sentence.split()]

    def convert_to_vectors_and_labels(self, sentence):
        """ expects single string like 'the quick brown fox jumped over the yellow house' 
            returns a list of vectors

            the final word in the sentence is considered the label
        """
        vectors = [self.word_vectors[word] for word in sentence.split()]
        return [vectors[:-1], vectors[-1]]

class TextRNN(object):

    def __init__(self, dim_in, dim_hidden, dim_out, **kwargs):

        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out

        self.input_layer = Linear(input_dim=self.dim_in, output_dim=self.dim_hidden,
                                weights_init=initialization.IsotropicGaussian(),
                                biases_init=initialization.Constant(0))
        self.input_layer.initialize()

        sparse_init = initialization.Sparse(num_init=15, weights_init=initialization.IsotropicGaussian())
        self.recurrent_layer = SimpleRecurrent(
                                dim=self.dim_hidden, activation=Tanh(), name="first_recurrent_layer",
                                weights_init=sparse_init,
                                biases_init=initialization.Constant(0.01))
        '''
        self.recurrent_layer = LSTM(dim=self.dim_hidden, activation=Tanh(),
                                    weights_init=initialization.IsotropicGaussian(std=0.001),
                                    biases_init=initialization.Constant(0.01))
        '''
        self.recurrent_layer.initialize()

        self.output_layer = Linear(input_dim=self.dim_hidden, output_dim=self.dim_out,
                                weights_init=initialization.Uniform(width=0.01),
                                biases_init=initialization.Constant(0.01))
        self.output_layer.initialize()

        self.children = [self.input_layer, self.recurrent_layer, self.output_layer]

    '''
    @recurrent(sequences=['inputs'], 
            states=['states'],
            contexts=[],
            outputs=['states', 'output'])
    '''

    def run(self, inputs):
        output = self.output_layer.apply( self.recurrent_layer.apply(self.input_layer.apply(inputs)) )
        return output

    #def get_dim(self, name):
    #    if name == 'states':
    #        return self.dim_hidden
    #    elif name == 'inputs':
    #        return self.dim_in
    #    elif name == 'output':
    #        return self.dim_out
    #    else:
    #        return super(TextRNN, self).get_dim(name)

words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 
            'plus', 'minus', 'prev', 'next', 'times'] 



test_sentences = [
                    'zero plus seven seven',
                    'zero plus six six',
                    'one plus six seven',
                    'one plus four five',
                    'two plus seven nine',
                    'three plus four seven',
                    'four plus six ten',
                    'five plus four nine',
                    'five plus five ten',
                    'six plus one six',
                    'six plus two seven',
                    'one minus seven six',
                    'two minus six four',
                    'three minus five two',
                    'six six next seven',
                    'two two prev one',
                    'one times five five',
                    'zero times eight zero',
                ]
shuffle(test_sentences)

sentences = [ 'zero plus zero zero',
                'one plus zero one',
                'zero plus one one',
                'zero plus two two',
                'zero plus three three',
                'zero plus four four',
                'zero plus five five',
                'zero plus eight eight',
                'zero plus nine nine',
                'zero plus ten ten',
                'one plus one two',
                'one plus two three',
                'one plus three four',
                'one plus five six',
                'one plus seven eight',
                'one plus eight nine',
                'one plus nine ten',
                'two plus one three',
                'two plus two four',
                'two plus three five',
                'two plus four six',
                'two plus five seven',
                'two plus six eight',
                'two plus eight ten',
                'three plus one four',
                'three plus two five',
                'three plus three six',
                'three plus five eight',
                'three plus six nine',
                'three plus seven ten',
                'four plus one five',
                'four plus two six',
                'four plus three seven',
                'four plus four eight',
                'four plus five nine',
                'five plus one six',
                'five plus two seven',
                'five plus three eight',
                'six plus three eight',
                'six plus four ten',
                'seven plus one eight',
                'seven plus two nine',
                'seven plus three ten',
                'eight plus one nine',
                'eight plus two ten',
                'nine plus one ten',
                'seven minus four three',
                'six minus three three',
                'five minus two three',
                'four minus four zero',


                'zero minus one one',
                'zero minus two two',
                'zero minus three three',
                'zero minus four four',
                'zero minus five five',
                'zero minus six six',
                'zero minus seven seven',
                'zero minus eight eight',
                'zero minus nine nine',
                'zero minus ten ten',
                'one minus one zero',
                'one minus two one',
                'one minus three two',
                'one minus four three',
                'one minus five four',
                'one minus six five',
                'one minus eight seven',
                'one minus nine eight',
                'two minus one one',
                'two minus two zero',
                'two minus three one',
                'two minus four two',
                'two minus five three',
                'two minus seven five',
                'two minus eight six',
                'three minus one two',
                'three minus two one',
                'three minus three zero',
                'three minus four one',
                'three minus six three',
                'three minus seven four',
                'four minus one three',
                'four minus two two',
                'four minus three one',
                'four minus five one',
                'four minus six two',
                'five minus one four',
                'five minus three two',
                'five minus four one',
                'five minus five zero',
                'six minus one five',
                'six minus two four',
                'six minus four two',
                'seven minus one six',
                'seven minus two five',
                'seven minus three four',
                'seven minus five two',
                'seven minus six one',
                'seven minus seven zero',
                'eight minus seven one',
                'eight minus six two',
                'eight minus five eight',
                'eight minus four four',
                'eight minus three five',
                'eight minus two six',
                'eight minus one seven',
                'eight minus zero eight',
                'nine minus zero nine',
                'nine minus one eight',
                'nine minus two seven',
                'nine minus three six',
                'nine minus four five',
                'nine minus five four',
                'nine minus six three',
                'nine minus seven two',
                'nine minus eight one',
                'nine minus nine zero',
                'nine minus ten one',


                'zero times one zero',
                'zero times two zero',
                'zero times three zero',
                'zero times four zero',
                'zero times five zero',
                'zero times six zero',
                'zero times seven zero',
                'zero times nine zero',
                'zero times ten zero',
                'one times one one',
                'one times two two',
                'one times three three',
                'one times four four',
                'one times six six',
                'one times seven seven',
                'one times eight eight',
                'one times nine nine',
                'two times one two',
                'two times two four',
                'two times three six',
                'two times four eight',
                'two times five ten',
                'three times one three',
                'three times two six',
                'three times three nine',
                'four times one four',
                'four times two eight',
                'five times one five',
                'five times two ten',
                'six times one six',
                'seven times one seven',
                'eight times one eight',
                'nine times one nine',


                'zero zero next one',
                'zero zero prev nine',
                'two two next three',
                'three three next four',
                'three three prev two',
                'four four next five',
                'four four prev three',
                'five five next six',
                'five five prev four',
                'six six prev five',
                'seven seven next eight',
                'seven seven prev six',
                'eight eight next seven',
                'eight eight prev nine',
                'nine nine next ten',
                'nine nine prev eight',
            ]
shuffle(sentences)

"""
sentences = [
                'three four five six',
                'one two three four',
                'seven eight nine ten',
                'four five six seven',
                'three four five six',
                'one two three four',
                'seven eight nine ten',
                'four five six seven',
                'three four five six',
                'one two three four',
                'seven eight nine ten',
                'four five six seven',
                'three four five six',
                'one two three four',
                'seven eight nine ten',
                'four five six seven'
            ]

test_sentences = [
                    'two three four five',
                    'five six seven eight',
                    'two three four five',
                    'five six seven eight',
                    'two three four five',
                    'five six seven eight',
                    'two three four five',
                    'five six seven eight',
                    'two three four five',
                    'five six seven eight',
                    'two three four five',
                    'five six seven eight',
                    'two three four five',
                    'five six seven eight',
                    'two three four five',
                    'five six seven eight'
                 ]
"""

# DATA SETUP
x = T.tensor.tensor3('x')
y = T.tensor.matrix('targets')

VECTOR_SIZE = 20
HIDDEN_UNITS = 100
RUN_N_TIMES = 1

word_bank = WordBank(words, VECTOR_SIZE)

for run_counter in range(RUN_N_TIMES):

    dataset = [word_bank.convert_to_vectors_and_labels(sentence) for sentence in sentences]
    test_dataset = [word_bank.convert_to_vectors_and_labels(sentence) for sentence in test_sentences]

    # MODEL SETUP
    textRNN = TextRNN(dim_in=VECTOR_SIZE, dim_hidden=HIDDEN_UNITS, dim_out=VECTOR_SIZE)

    output = textRNN.run(inputs=x)
    #get_states_and_output = T.function([x, x_mask], [output])

    # COST SETUP
    #y_hat = np.float32(np.ones((3,1)))
    labels = np.float32([data[1] for data in dataset])
    inputs_data = np.float32([data[0] for data in dataset])
    test_labels = np.float32([data[1] for data in test_dataset])
    test_inputs_data = np.float32([data[0] for data in test_dataset])

    cost = SquaredError().apply(y, output)
    cost.name = 'MSE_with_regularization'
    cg = ComputationGraph(cost)

    #inputs = VariableFilter(roles=[INPUT], bricks=[SimpleRecurrent])(cg.variables)
    #inputs = [inputs[0]]
    #cg_dropout = apply_dropout(cg, inputs, 0.5)
    #fprop_dropout = T.function([cg_dropout.inputs], [cg_dropout.outputs[0]])
    #dropped_out = VariableFilter(roles=[DROPOUT])(cg.variables)
    #inputs_referenced = [var.tag.replacement_of for var in dropped_out]
    #set(inputs) == set(inputs_referenced)

    get_states_and_output = T.function([x], [output])

    #W = VariableFilter(roles=[WEIGHT])(cg.variables)
    #W = W
    #cost = cost + 0.005 * (W ** 2).sum()

    sources = ('x', 'targets' )
    data = (inputs_data, labels)
    test_data = (test_inputs_data, test_labels)
    iterscheme = ShuffledScheme(len(inputs_data), batch_size=20)
    test_iterscheme = ShuffledScheme(len(test_inputs_data), batch_size=5)
    indexable_dataset = IndexableDataset(OrderedDict(zip(sources,data)))
    ds = DataStream( indexable_dataset, iteration_scheme=iterscheme )
    ds = Mapping(ds, swap01)
    test_indexable_dataset = IndexableDataset(OrderedDict(zip(sources,test_data)))
    test_ds = DataStream( test_indexable_dataset, iteration_scheme=test_iterscheme )
    test_ds = Mapping(test_ds, swap01)

    step_rule = Momentum(learning_rate=0.0001, momentum=0.7)
    #step_rule = RMSProp(learning_rate=0.001, decay_rate=0.7)
    algo = GradientDescent(cost=cost,
                            params=cg.parameters,
                            step_rule=step_rule)

    """
    results = get_states_and_output(np.swapaxes(inputs_data, 0, 1))
    for count in range(len(results[1])):
        question = ' '.join([word_bank.get_closest_words(vec,1)[0] for vec in inputs_data[count]])
        print 'INPUT: %s' % question
        answer = word_bank.get_closest_words( results[1][count], 1 )[0]
        print 'ANSWER: %s' % answer
        print '==========================================='
    """

    monitor = DataStreamMonitoring(variables=[cost],
                                    data_stream=test_ds,
                                    prefix='test')
    main_loop = MainLoop( 
                data_stream=ds,
                algorithm=algo,
                extensions=[FinishAfter(after_n_epochs=100), Printing()])

    main_loop.run()

    results = get_states_and_output(np.swapaxes(inputs_data, 0, 1))
    for count in range(10):
        question = ' '.join([word_bank.get_closest_words(vec,1)[0] for vec in inputs_data[count]])
        print 'INPUT: %s' % question
        answer = word_bank.get_closest_words( results[0][1][count], 1 )[0]
        print 'ANSWER: %s' % answer
        print '==========================================='

    print '==========================================='
    print '==========================================='
    print '             TEST DATA'
    print '==========================================='
    test_results = get_states_and_output(np.swapaxes(test_inputs_data, 0, 1))
    import pdb; pdb.set_trace()
    for count in range(10):
        try:
            question = ' '.join([word_bank.get_closest_words(vec,1)[0] for vec in test_inputs_data[count]])
        except:
            import pdb; pdb.set_trace()
        print 'INPUT: %s' % question
        answer = word_bank.get_closest_words( test_results[1][count], 1 )[0]
        print 'ANSWER: %s' % answer
        print 'weight norms %.2f / %.2f / %.2f' % (norm(test_results[0][0][count]), norm(test_results[0][1][count]),
                                                norm(test_results[0][2][count]))
        print '==========================================='
        
    for result_idx in range(len(results[1])):
        word_bank.set_nearest_vector(labels[result_idx], results[1][result_idx])
