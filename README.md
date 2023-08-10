# Generic-ChatBot

Environment Setup:

Make sure you have Python installed on your machine.
Install required packages using pip install numpy tensorflow nltk.
Prepare the Data:

Create an intents.json file with your chatbot intents.
Run the new.py script to preprocess the data and create vocabulary files (words.pkl and classes.pkl).
Train the Model:

Run the chatbot.py script to train the neural network model on your data.
The model will be saved as chatbot_model.h5.
Preprocess Input:

Define the preprocess_input function in your code to tokenize and preprocess user input.
Get Responses:

Implement the get_response function to process user input and provide appropriate responses using the trained model.
Chatbot Interaction:

Run the interaction loop using a while loop.
Input 'exit' to exit the loop and end the conversation.
