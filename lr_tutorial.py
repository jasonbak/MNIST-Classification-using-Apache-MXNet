import mxnet as mx
from mxnet import nd, autograd, gluon

# Set the context
data_ctx = mx.cpu()
model_ctx = mx.cpu()


# Help get train data in the final form
num_inputs = 1
num_examples = 1000
raw_data = nd.random.uniform(low=-1, high=1, shape=(num_examples,
    num_inputs))

# Set up training data
num_features = 2
raw_train_data = nd.ones((num_examples, num_features), ctx=data_ctx)
raw_train_data[:,0] = (raw_data[:,0]**4)
raw_train_data[:,1] = (raw_data[:,0])
train_noise = 0.5*nd.random.normal(shape=(num_examples,), ctx=data_ctx)

def actualFunc(raw_train_data):
    return 5*raw_train_data[:, 0] + 3.5*raw_train_data[:, 1] + 3
train_label = actualFunc(raw_train_data) + train_noise

# Create data iterators
train_dataset = gluon.data.ArrayDataset(raw_train_data, train_label)
batch_size = 32
train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model and define the model's architecture
net = gluon.nn.Dense(1)

# Initialize model's parameters from a Normal distribution with std dev of 0.05
net.collect_params().initialize(mx.init.Normal(sigma=0.05), ctx=model_ctx)

# Define loss function (how well the model is able to correctly predict)
square_loss = gluon.loss.L2Loss()

# Use SGD training algorithm
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.1})

epochs = 10
for e in range(epochs):
    cumulative_loss = 0
    for data, label in train_data:
        with autograd.record(): # Start recording gradients
            # Generate predictions on the forward pass
            output = net(data)
            # Calcualte loss
            loss = square_loss(output, label)
            # Perform backprop
            loss.backward()
        # Update parameters
        trainer.step(batch_size)
        cumulative_loss += nd.mean(loss).asscalar()
    # After each epoch, report loss on training
    print("Epoch %s, loss: %s" % (e+1, cumulative_loss / num_examples))

# Retrieve learned model's parameters
params = net.collect_params()
for param in params.values():
    print(param.name,param.data())
