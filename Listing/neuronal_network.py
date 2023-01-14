model = Sequential()

X, y = utils.prepare_for_tensor(X_data, target)

# 12 input nodes
model.add(Dense(12, input_dim=X_data.shape[1], activation='sigmoid', name='input'))

# 20 nodes for the hidden layers
model.add(Dense(20, activation='relu', name='layer1'))
model.add(Dense(20, activation='relu', name='layer2'))
model.add(Dense(20, activation='relu', name='layer3'))
model.add(Dense(1, activation='sigmoid', name='output'))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

