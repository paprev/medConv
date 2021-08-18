from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import sklearn
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt

EPOCHS = 10
BATCH_SIZE = 10
N_HIDDEN = 768
N_HIDDEN_2 = 768
EMB = 768
MAX_LEN = 128

X = np.load('word_bert.npy')
print('X.shape',X.shape)
# X = np.load('sent_bert.npy')
Y = np.load('lab_sent.npy')
# X_o = np.load('sent_bert.npy')
X_o = np.load('word_bert.npy')
# X_d = np.load('sent_bert.npy')
X_d = np.load('word_bert.npy')
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
N_CLASS = 2
x_train, x_valid, y_train, y_valid = train_test_split(X,encoded_Y, test_size=0.1, random_state=42)
x_train_o, x_valid_o, y_train_o, y_valid_o = train_test_split(X_o,encoded_Y, test_size=0.1, random_state=42)
x_train_d, x_valid_d, y_train_d, y_valid_d = train_test_split(X_d,encoded_Y, test_size=0.1, random_state=42)
print('EMB',EMB)
model = My(N_HIDDEN, N_HIDDEN_2, EMB, MAX_LEN)
model.compile('adam', 'binary_crossentropy', metrics=['acc'])
# model.fit(x_train, y_train, validation_data=[x_valid, y_valid],epochs=EPOCHS, batch_size=BATCH_SIZE)
model.summary()
# sys.exit()
# print('in fit!')
history = model.fit([x_train,x_train_o,x_train_d], to_categorical(y_train, 2), epochs=EPOCHS, batch_size=BATCH_SIZE)
x_test = x_valid
y_test = y_valid
x_test_o = x_valid_o
y_test_o = y_valid_o
x_test_d = x_valid_d
y_test_d = y_valid_d

# results = test_model(model, x_test, y_test)
# print('results', results)
# print('np.asarray(results).shape', np.asarray(results).shape)
# print('x_test.shape', x_test.shape)

pred = model.predict([x_test,x_test_o,x_test_d], verbose=1)
print('pred.shape', pred.shape)
print('pred',pred)
print('y_test',y_test)
pred_n = np.argmax(pred, axis=1)
print('pred_n', pred_n)
precisions, recall, f1_score, _ = precision_recall_fscore_support(y_test, pred_n, average='weighted')
precisions_mi, recall_mi, f1_score_mi, _ = precision_recall_fscore_support(y_test, pred_n, average='micro')
precisions_ma, recall_ma, f1_score_ma, _ = precision_recall_fscore_support(y_test, pred_n, average='macro')
acc = accuracy_score(y_test, pred_n)
print('precisions',precisions)
print('recall',recall)
print('f1_score',f1_score)
print('precisions_mi',precisions_mi)
print('recall_mi',recall_mi)
print('f1_score_mi',f1_score_mi)
print('precisions_ma',precisions_ma)
print('recall_ma',recall_ma)
print('f1_score_ma',f1_score_ma)
print('acc',acc)