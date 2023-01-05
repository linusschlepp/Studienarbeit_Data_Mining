model = Sequential()

X, y = utils.prepare_for_tensor(X_data, target)


model.add(Dense(14, input_dim=X_data.shape[1], activation='sigmoid', name='input'))


model.add(Dense(15, activation='relu', name='layer1'))
model.add(Dense(15, activation='relu', name='layer2'))
model.add(Dense(15, activation='relu', name='layer2'))
model.add(Dense(1, activation='sigmoid', name='output'))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

