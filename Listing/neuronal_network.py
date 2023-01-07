model = Sequential()

X, y = utils.prepare_for_tensor(X_data, target)

# 9 features = 9 input nodes 
model.add(Dense(9, input_dim=X_data.shape[1], activation='sigmoid', name='input'))

# 15 nodes for the hidden layers
model.add(Dense(15, activation='relu', name='layer1'))
model.add(Dense(15, activation='relu', name='layer2'))
model.add(Dense(15, activation='relu', name='layer2'))
model.add(Dense(1, activation='sigmoid', name='output'))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

