import numpy as np
import random
import copy
import tensorflow as tf
import operator 
from .variables import (interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars,
                        VariableState, eigvector_vars)
import gc
class IER:

    def __init__(self, session, variables=None, transductive=False, pre_step_op=None):
        self.session = session
        self._model_state = VariableState(self.session, variables or tf.trainable_variables())
        self._full_state = VariableState(self.session,
                                         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        self._transductive = transductive
        self._pre_step_op = pre_step_op



    def train_step_Eigen(self,
                   dataset,
                   input_ph,
                   label_ph,
                   minimize_op,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   replacement,
                   meta_step_size,
                   meta_batch_size):
        old_vars_o = self._model_state.export_variables()
        new_vars = []
        v_length = 0
        temp = []
        for para in self._model_state.export_variables():
            para = np.array(para)
            temp = np.r_[temp,para.flatten()]

        for _ in range(meta_batch_size):
            # old_vars = copy.deepcopy(old_vars_o)
            w_r = []
            m_k = 0

            mini_dataset = _sample_mini_dataset(dataset, num_classes, num_shots)
            for batch in _mini_batches(mini_dataset, inner_batch_size, inner_iters, replacement):
                weights_li = []
                m_k += 1
                inputs, labels = zip(*batch)

                if self._pre_step_op:
                    self.session.run(self._pre_step_op)
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
                # print(self.session.run(lp, feed_dict={input_ph: inputs, label_ph: labels}))
                # print(self.session.run(v, feed_dict={input_ph: inputs, label_ph: labels}))

                for para in self._model_state.export_variables():
                    para = np.array(para)
                    weights_li = np.r_[weights_li,para.flatten()]
                w_r = np.r_[w_r,weights_li]
            w_r = w_r.reshape( m_k , int(len(w_r) / m_k))
            # jilu = np.vstack((temp,w_r))

            w_r = np.vstack((temp,w_r))
            m_k = m_k + 1
            jilu = copy.deepcopy(w_r)
            a , b , c , q , p , matrix = get_eigen(w_r)
            del w_r
            eigval, eigvector, sum_eigval = process_eigvector( a, b, c , q, p, matrix )
            del matrix
            le = 0

            for i in range( m_k - 1 ):
                v_b = jilu[i+1] - jilu[i]

                le += np.abs( np.dot(v_b,eigvector.T[0])) 
            #     le += np.dot(v_b,eigvector.T[0])
            # le = np.abs( le )


            v_d = np.mean(jilu[int(m_k/2):] - jilu[0:int(m_k/2)],axis=0)
            del jilu
            gc.collect()
            if get_direction(eigvector[:,0],v_d) == False:
                eigvector[:,0] = -eigvector[:,0]
            new_vars.append(    eigval[0] / sum_eigval * eigvector[:,0])
            v_length += le
            self._model_state.import_variables(old_vars_o)     

        new_vars = np.mean( new_vars,axis = 0 )
        self._model_state.import_variables( eigvector_vars( old_vars_o, new_vars,  v_length / meta_batch_size * meta_step_size ))
        # self._model_state.import_variables( eigvector_vars( old_vars_o, new_vars,   meta_step_size / meta_batch_size ))
 
    
   
    def train_step_Eigen_O(self,
                   dataset,
                   input_ph,
                   label_ph,
                   minimize_op,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   replacement,
                   meta_step_size,
                   meta_batch_size):
        old_vars_o = self._model_state.export_variables()
        new_vars = []
        v_length = 0
        temp = []
        for para in self._model_state.export_variables():
            para = np.array(para)
            temp = np.r_[temp,para.flatten()]

        for _ in range(meta_batch_size):
            # old_vars = copy.deepcopy(old_vars_o)
            w_r = []
            m_k = 0

            mini_dataset = _sample_mini_dataset(dataset, num_classes, num_shots)
            for batch in _mini_batches(mini_dataset, inner_batch_size, inner_iters, replacement):
                weights_li = []
                m_k += 1
                inputs, labels = zip(*batch)
                if self._pre_step_op:
                    self.session.run(self._pre_step_op)
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})

                for para in self._model_state.export_variables():
                    para = np.array(para)
                    weights_li = np.r_[weights_li,para.flatten()]
                w_r = np.r_[w_r,weights_li]
            w_r = w_r.reshape( m_k , int(len(w_r) / m_k))
            # jilu = np.vstack((temp,w_r))

            w_r = np.vstack((temp,w_r))
            m_k = m_k + 1
            jilu = copy.deepcopy(w_r)
            a , b , c , q , p , matrix = get_eigen(w_r)
            eigval, eigvector, sum_eigval = process_eigvector( a, b, c , q, p, matrix )

            le = 0

            for i in range( m_k - 1 ):
                v_b = jilu[i+1] - jilu[i]

                le += np.abs( np.dot(v_b,eigvector.T[0])) 
            #     le += np.dot(v_b,eigvector.T[0])
            # le = np.abs( le )


            v_d = np.mean(jilu[int(m_k/2):] - jilu[0:int(m_k/2)],axis=0)

            if get_direction(eigvector[:,0],v_d) == False:
                eigvector[:,0] = -eigvector[:,0]
            new_vars.append(    eigval[0] / sum_eigval * eigvector[:,0])
            v_length += le
            self._model_state.import_variables(old_vars_o)     

        new_vars = np.mean( new_vars,axis = 0 )
        self._model_state.import_variables( eigvector_vars( old_vars_o, new_vars,  v_length / meta_batch_size * meta_step_size ))
        # self._model_state.import_variables( eigvector_vars( old_vars_o, new_vars,   meta_step_size / meta_batch_size ))



    def evaluate(self,
                 dataset,
                 input_ph,
                 label_ph,
                 minimize_op,
                 predictions,
                 num_classes,
                 num_shots,
                 inner_batch_size,
                 inner_iters,
                 replacement):
        """
        Run a single evaluation of the model.

        Samples a few-shot learning task and measures
        performance.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          predictions: a Tensor of integer label predictions.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.

        Returns:
          The number of correctly predicted samples.
            This always ranges from 0 to num_classes.
        """
        train_set, test_set = _split_train_test(
            _sample_mini_dataset(dataset, num_classes, num_shots+1))
        old_vars = self._full_state.export_variables()
        for batch in _mini_batches(train_set, inner_batch_size, inner_iters, replacement):
            inputs, labels = zip(*batch)
            if self._pre_step_op:
                self.session.run(self._pre_step_op)
            self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
        test_preds = self._test_predictions(train_set, test_set, input_ph, predictions)
        num_correct = sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)])
        self._full_state.import_variables(old_vars)
        return num_correct

    def _test_predictions(self, train_set, test_set, input_ph, predictions):
        if self._transductive:
            inputs, _ = zip(*test_set)
            return self.session.run(predictions, feed_dict={input_ph: inputs})
        res = []
        for test_sample in test_set:
            inputs, _ = zip(*train_set)
            inputs += (test_sample[0],)
            res.append(self.session.run(predictions, feed_dict={input_ph: inputs})[-1])
        return res


def _sample_mini_dataset(dataset, num_classes, num_shots):
    """
    Sample a few shot task from a dataset.

    Returns:
      An iterable of (input, label) pairs.
    """
    shuffled = list(dataset)
    random.shuffle(shuffled)
    for class_idx, class_obj in enumerate(shuffled[:num_classes]):
        for sample in class_obj.sample(num_shots):
            yield (sample, class_idx)

def _mini_batches(samples, batch_size, num_batches, replacement):
    """
    Generate mini-batches from some data.

    Returns:
      An iterable of sequences of (input, label) pairs,
        where each sequence is a mini-batch.
    """
    samples = list(samples)
    if replacement:
        for _ in range(num_batches):

            yield random.sample(samples, batch_size)
        return
    cur_batch = []
    batch_count = 0
    while True:
        random.shuffle(samples)
        for sample in samples:
            cur_batch.append(sample)
            if len(cur_batch) < batch_size:
                continue
            yield cur_batch
            cur_batch = []
            batch_count += 1
            if batch_count == num_batches:
                return

def _split_train_test(samples, test_shots=1):
    """
    Split a few-shot task into a train and a test set.

    Args:
      samples: an iterable of (input, label) pairs.
      test_shots: the number of examples per class in the
        test set.

    Returns:
      A tuple (train, test), where train and test are
        sequences of (input, label) pairs.
    """
    train_set = list(samples)
    test_set = []
    labels = set(item[1] for item in train_set)
    for _ in range(test_shots):
        for label in labels:
            for i, item in enumerate(train_set):
                if item[1] == label:
                    del train_set[i]
                    test_set.append(item)
                    break
    if len(test_set) < len(labels) * test_shots:
        raise IndexError('not enough examples of each class for test set')
    return train_set, test_set


def get_eigen(matrix):

    # print(list(matrix.shape))
    m_ = np.mean(matrix, axis=0)
    q , p = list(matrix.shape)
    # std =  np.std(matrix , axis = 0)
    # print(len(std))
    matrix -= m_.reshape(1, p).repeat(q, 0)
    # matrix /= np.max(np.abs(matrix))
    # matrix /= (np.max(np.abs(matrix) , axis = 0) - np.min(np.abs(matrix) , axis = 0) )

    # matrix /= np.max(np.abs(matrix) , axis=0)
    # mask = std < 1e-8
    # stdd = std + mask
    # matrix /= stdd
    # matrix[:,mask] = 0

   
    A = np.dot( matrix , matrix.T )
    a , b = np.linalg.eig(A)
    return a , b , m_ , q , p , matrix
    # except:
    #     return 1
    # A = np.dot( matrix , matrix.T )
    # a , b = np.linalg.eig(A)
    
    # return a , b , m_ , q , p

def process_eigvector(a,b,m_,q,p,w_r):
    sum_eigval = sum(a)
    # a , b = get_eigen(w_r)
    eigvector = np.dot(w_r.T,b)
    # print(eigvector.shape)
    eigValInd = np.argsort(-a) 
    # eigValInd = eigValInd[:]
    redEigVects = eigvector[:,eigValInd] 
    eigvector = redEigVects
    eig_norm = np.linalg.norm(eigvector , axis=0)
    eigvector /= eig_norm.reshape(1,q)
    eigval = a[eigValInd]

    return eigval, eigvector, sum_eigval

def get_direction(v_1,v_2):
    return np.dot(v_1,v_2)>0


def index_lst(lst, rate):

    for i in range(1, len(lst)):
        if sum(lst[:i])/sum(lst) >= rate:
            return i


def get_mean(matrix,n):
    sum_ = np.sum(matrix , axis = 0)
    return sum_ / n



def flip_label(y, ratio,  pattern='sym', one_hot=False,n_class=5):
    #y: true label, one hot
    #pattern: 'pair' or 'sym'
    #p: float, noisy ratio
    
    #convert one hot label to int
    y = np.array(y)
    if one_hot:
        y = np.argmax(y,axis=1)#[np.where(r==1)[0][0] for r in y]
    # n_class = max(y)+1
    
    #filp label
    for i in range(len(y)):
        if pattern=='sym':
            p1 = ratio/(n_class-1)*np.ones(n_class)
            p1[y[i]] = 1-ratio
            y[i] = np.random.choice(n_class,p=p1)
        elif pattern=='asym':
            y[i] = np.random.choice([y[i],(y[i]+1)%n_class],p=[1-ratio,ratio])            
            
    #convert back to one hot
    if one_hot:
        y = np.eye(n_class)[y]
    y = tuple(y)
    return y


def split_(dataset, num_train):
    """
    Split the dataset into a training and test set.

    Args:
      dataset: an iterable of Characters.

    Returns:
      A tuple (train, test) of Character sequences.
    """
    all_data = list(dataset)
    random.shuffle(all_data)

    return all_data[:num_train], all_data[num_train:]







def _mini_batches_2(samples, batch_size, num_batches, replacement):
    """
    Generate mini-batches from some data.

    Returns:
      An iterable of sequences of (input, label) pairs,
        where each sequence is a mini-batch.
    """
    samples = list(samples)

    if replacement:
        for _ in range(num_batches):

            yield random.sample(samples, batch_size)
        return

    cur_batch = []
    batch_count = 0
    while True:

        for sample in samples:
            cur_batch.append(sample)
            if len(cur_batch) < batch_size:

                continue
            yield cur_batch

            cur_batch = []
            batch_count += 1
            if batch_count == num_batches:
                return

