import mxnet as mx
from mxnet import gluon, autograd, nd

# Set the context
data_ctx = mx.cpu()
model_ctx = mx.cpu()

# Import the dataset using a native MXNet utils function
raw_data = mx.test_utils.get_mnist()

# We have 28*28=784 pixels; flatten all the pixels
# Set up the training data and reshape the pictures
raw_train_data_np = raw_data['train_data'].reshape((-1, 784))
raw_train_data = mx.nd.array(raw_train_data_np, ctx=data_ctx)
train_label_np = mx.nd.array(raw_data['train_label'])
train_label = mx.nd.array(train_label_np, ctx=data_ctx)
num_examples = raw_train_data.shape[0]

# Set up the test data and reshape the pictures
test_data_np = raw_data['test_data'].reshape((-1, 784))
test_data = mx.nd.array(test_data_np)
test_label_np = raw_data['test_label']
test_label = mx.nd.array(test_label_np, ctx=data_ctx)

train_dataset = gluon.data.ArrayDataset(raw_train_data, train_label)
batch_size = 32
train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# train_data = mx.io.NDArrayIter(raw_train_data, label=train_label,batch_size=batch_size, shuffle=True)

# Initialize the model
net = gluon.nn.Sequential()
# Define the model's architecture
num_hidden = 64
with net.name_scope():
    # 1st layer
    net.add(gluon.nn.Dense(num_hidden, activation="sigmoid"))
    # 2nd layer
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    # Output layer
    net.add(gluon.nn.Dense(10))

# Initialize model's parameters from a Normal distribution with std dev of 0.05
net.collect_params().initialize(mx.init.Normal(sigma=0.05), ctx=model_ctx)

# Define loss function (how well the model is able to correctly predict)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# Use SGD training algorithm
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.1})

epochs = 10
for e in range(epochs):
    cumulative_loss = 0
    for data,label in train_data:
        with autograd.record(): # Start recording gradients
            # Generate predictions on the the forward pass
            output = net(data)
            # Calculate loss
            loss = softmax_cross_entropy(output, label)
            # Perform backprop
            loss.backward()
        # Update parameters
        trainer.step(batch_size)
        cumulative_loss += nd.mean(loss).asscalar()
    # After each epoch, report loss on training
    print("Epoch %s, loss: %s" % (e+1, cumulative_loss / num_examples))

# Calculate and report accuracy of predictions from our learned model
acc = mx.metric.Accuracy()
output = net(test_data)
predictions = nd.argmax(output, axis=1)
acc.update(preds=predictions, labels=test_label)
print(acc)
