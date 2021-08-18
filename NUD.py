# from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.merge import concatenate
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as K
from keras.engine.topology import Layer
from keras.layers.advanced_activations import Softmax
from keras.activations import linear

class Similarity(Layer):

    def __init__(self, **kwargs):
        super(Similarity, self).__init__(**kwargs)

    def compute_similarity(self, repeated_context_vectors, repeated_query_vectors):
        element_wise_multiply = repeated_context_vectors * repeated_query_vectors
        concatenated_tensor = K.concatenate(
            [repeated_context_vectors, repeated_query_vectors, element_wise_multiply], axis=-1)
        dot_product = K.squeeze(K.dot(concatenated_tensor, self.kernel), axis=-1)
        return linear(dot_product + self.bias)

    def build(self, input_shape):
        word_vector_dim = input_shape[0][-1]
        weight_vector_dim = word_vector_dim * 3
        self.kernel = self.add_weight(name='similarity_weight',
                                      shape=(weight_vector_dim, 1),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='similarity_bias',
                                    shape=(),
                                    initializer='ones',
                                    trainable=True)
        super(Similarity, self).build(input_shape)

    def call(self, inputs):
        context_vectors, query_vectors = inputs
        num_context_words = K.shape(context_vectors)[1]
        num_query_words = K.shape(query_vectors)[1]
        context_dim_repeat = K.concatenate([[1, 1], [num_query_words], [1]], 0)
        query_dim_repeat = K.concatenate([[1], [num_context_words], [1, 1]], 0)
        repeated_context_vectors = K.tile(K.expand_dims(context_vectors, axis=2), context_dim_repeat)
        repeated_query_vectors = K.tile(K.expand_dims(query_vectors, axis=1), query_dim_repeat)
        similarity_matrix = self.compute_similarity(repeated_context_vectors, repeated_query_vectors)
        return similarity_matrix

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        num_context_words = input_shape[0][1]
        num_query_words = input_shape[1][1]
        return (batch_size, num_context_words, num_query_words)

    def get_config(self):
        config = super().get_config()
        return config

class C2QAttention(Layer):

    def __init__(self, **kwargs):
        super(C2QAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(C2QAttention, self).build(input_shape)

    def call(self, inputs):
        similarity_matrix, encoded_question = inputs
        context_to_query_attention = Softmax(axis=-1)(similarity_matrix)
        encoded_question = K.expand_dims(encoded_question, axis=1)
        return K.sum(K.expand_dims(context_to_query_attention, axis=-1) * encoded_question, -2)

    def compute_output_shape(self, input_shape):
        similarity_matrix_shape, encoded_question_shape = input_shape
        return similarity_matrix_shape[:-1] + encoded_question_shape[-1:]

    def get_config(self):
        config = super().get_config()
        return config

class Q2CAttention(Layer):

    def __init__(self, **kwargs):
        super(Q2CAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Q2CAttention, self).build(input_shape)

    def call(self, inputs):
        similarity_matrix, encoded_context = inputs
        max_similarity = K.max(similarity_matrix, axis=-1)
        # by default, axis = -1 in Softmax
        context_to_query_attention = Softmax()(max_similarity)
        weighted_sum = K.sum(K.expand_dims(context_to_query_attention, axis=-1) * encoded_context, -2)
        expanded_weighted_sum = K.expand_dims(weighted_sum, 1)
        num_of_repeatations = K.shape(encoded_context)[1]
        return K.tile(expanded_weighted_sum, [1, num_of_repeatations, 1])

    def compute_output_shape(self, input_shape):
        similarity_matrix_shape, encoded_context_shape = input_shape
        return similarity_matrix_shape[:-1] + encoded_context_shape[-1:]

    def get_config(self):
        config = super().get_config()
        return config

class MergedContext(Layer):

    def __init__(self, **kwargs):
        super(MergedContext, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MergedContext, self).build(input_shape)

    def call(self, inputs):
        encoded_context, context_to_query_attention, query_to_context_attention = inputs
        element_wise_multiply1 = encoded_context * context_to_query_attention
        element_wise_multiply2 = encoded_context * query_to_context_attention
        concatenated_tensor = K.concatenate(
            [encoded_context, context_to_query_attention, element_wise_multiply1, element_wise_multiply2], axis=-1)
        return concatenated_tensor

    def compute_output_shape(self, input_shape):
        encoded_context_shape, _, _ = input_shape
        return encoded_context_shape[:-1] + (encoded_context_shape[-1] * 4, )

    def get_config(self):
        config = super().get_config()
        return config
def compute_similarity(repeated_context_vectors, repeated_query_vectors):
  element_wise_multiply = repeated_context_vectors * repeated_query_vectors
  concatenated_tensor = K.concatenate(
      [repeated_context_vectors, repeated_query_vectors, element_wise_multiply], axis=-1)
  # dot_product = K.squeeze(K.dot(concatenated_tensor, self.kernel), axis=-1)
  dot_product = K.squeeze(concatenated_tensor, axis=-1)
  # return linear(dot_product + self.bias)
  return linear(dot_product)

def My(hid=768, hid_1=1, emb=768, max_len=128):
  # first input model
  print('emb',emb)
  print('max_len',max_len)
  visible1 = kl.Input(shape=(max_len,emb))
  print('visible1.shape',visible1.shape)
  # second input model
  visible2 = kl.Input(shape=(max_len,emb))
  print(' visible2.shape', visible2.shape)
  #ontology-aware sentence representation
  conc_1 = kl.concatenate([visible1,visible2])
  print('conc_1.shape',conc_1.shape)
  dense_1 = kl.Dense(hid, activation='relu')
  ont_temp = kl.TimeDistributed(dense_1)(conc_1)
  # ont_temp = kl.Dense(hid, activation='relu')(conc_1)
  print('ont_temp.shape',ont_temp.shape)
  ont_repr = ont_temp * visible1
  print('ont_repr.shape',ont_repr.shape)
  # third input model
  visible3 = kl.Input(shape=(max_len,emb))
  #Cross-Attention Layer based on BiDAF model
  # similarity_matrix = Similarity(name='similarity_layer')([ont_repr, visible3])
  context_vectors, query_vectors = [ont_repr, visible3]
  print('context_vectors.shape',context_vectors.shape)
  print('query_vectors.shape',query_vectors.shape)
  num_context_words = K.shape(context_vectors)[1]
  print('num_context_words',num_context_words)
  num_query_words = K.shape(query_vectors)[1]
  print('num_query_words',num_query_words)
  context_dim_repeat = K.concatenate([[1, 1], [num_query_words], [1]], 0)
  print('context_dim_repeat.shape',context_dim_repeat.shape)
  query_dim_repeat = K.concatenate([[1], [num_context_words], [1, 1]], 0)
  print('query_dim_repeat.shape',query_dim_repeat.shape)
  repeated_context_vectors = K.tile(K.expand_dims(context_vectors, axis=2), context_dim_repeat)
  print('repeated_context_vectors.shape',repeated_context_vectors.shape)
  repeated_query_vectors = K.tile(K.expand_dims(query_vectors, axis=1), query_dim_repeat)
  print('repeated_query_vectors.shape',repeated_query_vectors.shape)
  # similarity_matrix = compute_similarity(repeated_context_vectors, repeated_query_vectors)
  # print('similarity_matrix.shape',similarity_matrix.shape)

  element_wise_multiply = repeated_context_vectors * repeated_query_vectors
  print('element_wise_multiply.shape',element_wise_multiply.shape)
  concatenated_tensor = K.concatenate(
      [repeated_context_vectors, repeated_query_vectors, element_wise_multiply], axis=-1)
  print('concatenated_tensor.shape',concatenated_tensor.shape)
  # input_shape = context_vectors.shape
  word_vector_dim = emb
  weight_vector_dim = word_vector_dim * 3
  # kernel = self.add_weight(name='similarity_weight',
  #                               shape=(weight_vector_dim, 1),
  #                               initializer='uniform',
  #                               trainable=True)
  w_init = tf.random_normal_initializer()
  kernel = tf.Variable(
            initial_value=w_init(shape=(weight_vector_dim, 1)),
            trainable=True,)
  # bias = self.add_weight(name='similarity_bias',
  #                             shape=(),
  #                             initializer='ones',
  #                             trainable=True)
  w_init = tf.ones_initializer()
  bias = tf.Variable(
            initial_value=w_init(shape=()),
            trainable=True,)
  dot_product = K.squeeze(K.dot(concatenated_tensor, kernel), axis=-1)
  print('dot_product.shape',dot_product.shape)
  similarity_matrix = linear(dot_product + bias)
  print('similarity_matrix.shape',similarity_matrix.shape)

  context_to_query_attention = C2QAttention(name='context_to_query_attention')([
      similarity_matrix, ont_repr])
  print('context_to_query_attention.shape',context_to_query_attention.shape)
  query_to_context_attention = Q2CAttention(name='query_to_context_attention')([
      similarity_matrix, visible3])
  print('query_to_context_attention.shape',query_to_context_attention.shape)

  merged_context = MergedContext(name='merged_context')(
      [visible3, context_to_query_attention, query_to_context_attention])
  print('merged_context.shape',merged_context.shape)
  new_repr = kl.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=False))(merged_context)
  print('new_repr.shape',new_repr.shape)
  # out3_1 = kl.Dense(hid, activation='relu')(visible3)
  # merge input models
  # merge = kl.concatenate([out1_1, out2_1, out3_1])
  # print('merge.shape',merge.shape)
  # interpretation model
  # hidden1 = Dense(10, activation='relu')(merge)
  # hidden2 = Dense(10, activation='relu')(hidden1)
  # output = kl.Dense(1, activation='sigmoid')(merge)
  output = kl.Dense(2, activation='softmax')(new_repr)
  print('output.shape',output.shape)
  model = Model(inputs=[visible1, visible2, visible3], outputs=output)
  # summarize layers
  # print(model.summary())
  # plot graph
  return model