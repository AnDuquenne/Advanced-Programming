# Advanced-Programming
We do this because we are using a technique called “teacher forcing” during training. Teacher forcing is a method used in sequence-to-sequence models, where the true output sequence is fed as input to the model during training instead of using the model’s own predictions from the previous time step. This helps the model to learn faster and more accurately.

{batch, features, time}
[0, 30] -> [30, 35]
[0, 30] -> [5, 35]



features = ["price", "volume", "open", "high", "low", "close", "returns"]

targets tensor([[2373.5000],
        [2373.2000],
        [2373.7100],
        [2373.9199],
        [2374.7000],
        [2373.0601],
        [2375.0000],
        [2374.7000],
        [2374.9900],
        [2376.3101]])