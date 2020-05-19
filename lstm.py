from keras.preprocessing import sequence, text
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM

#train_data_source = './exercise_data/human_dna_train_small.csv'
train_data_source = './exercise_data/human_dna_train_split.csv'
val_data_source = './exercise_data/human_dna_validation_split.csv'
test_data_source = './exercise_data/human_dna_test_split.csv'

train_df = pd.read_csv(train_data_source, header=0)
test_df = pd.read_csv(test_data_source, header=0)

train_seq = train_df['sequences']
train_label = train_df['labels']
test_seq = test_df['sequences']
test_label = test_df['labels']

# Preprocess 
tk = text.Tokenizer(char_level=True)
tk.fit_on_texts(train_seq)

train_seq_tok = tk.texts_to_sequences(train_seq)
test_seq_tok = tk.texts_to_sequences(test_seq)

train_seq = np.array(train_seq)
train_seq_tok = np.array(train_seq_tok)
train_label = np.array(train_label)
test_seq = np.array(test_seq)
test_seq_tok = np.array(test_seq_tok)
test_label = np.array(test_label)

print('train_seq shape:', train_seq.shape)
print('test_seq shape:', test_seq.shape)

###################################
############ LSTM Model ###########
###################################
max_features = 20000
batch_size = 32
epochs = 15
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
model.fit(train_seq_tok, train_label, batch_size=batch_size, epochs=epochs, validation_data=(test_seq_tok, test_label))
score, acc = model.evaluate(test_seq_tok, test_label, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)