from flask import Flask, render_template, request
import joblib
import os
# from model.chatbotmodel import ChatbotModel

app = Flask(__name__)


class ChatbotModel:
    def __init__(self, model_path='model/SVC_model.pkl', vectorizer_path='model/vectorizer.pkl'):
        # Get the absolute paths to the model and vectorizer files
        model_path = os.path.join(os.path.dirname(__file__), model_path)
        vectorizer_path = os.path.join(os.path.dirname(__file__), vectorizer_path)

        # Load the pre-trained model and vectorizer
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict_intent(self, user_input):
        # Vectorize the user input
        user_input_vec = self.vectorizer.transform([user_input])

        # Predict the intent
        intent = self.model.predict(user_input_vec)[0]

        return intent

    def generate_response(self, intent):
        # Implement your logic here to generate appropriate responses based on the predicted intents
        if intent == 'alcoholism':
            response = "I am sorry for what you are going through. Dealing with alcoholism can be a big challenge for everyone. However, you can always get help to get out of the addiction. Do you mind sharing with me what kind of help you would want?"
        elif intent == 'diagnosedPTSD':
            response = "Goodbye! Take care."
        elif intent == 'socialanxiety':
            response = "I'm sorry, I don't have the information you're looking for."
        else:
            response = "I'm here to help. Please let me know how I can assist you."

        return response

chatbot = ChatbotModel()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response_route():
    user_text = request.args.get('msg')
    response = chatbot.generate_response(user_text)
    return response

if __name__ == "__main__":
    app.run(debug=True, port=8000)
