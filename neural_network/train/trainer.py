def train(model, X, Y,loss_fn, epochs, batch_size, learning_rate):
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, X.shape[0], batch_size):
            x_batch = X[i:i+batch_size]
            y_batch = Y[i:i+batch_size]

            logits = model.forward(x_batch)
            loss = loss_fn.forward(logits, y_batch)
            grad = loss_fn.backward()

            model.backward(grad, learning_rate)
            epoch_loss += loss

        print(f"Epoch {epoch+1}, Loss: {epoch_loss/X.shape[0]:.4f}, Accuracy: {model.evaluate(X, Y):.4f}")
