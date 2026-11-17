from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Load the BERT sentiment model from Hugging Face (at startup)
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()


def analyze_sentiment(text):
    # Text -> token IDs (truncated to the model's 512-token limit)
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    # Forward pass without gradient tracking (inference only)
    with torch.no_grad():
        outputs = model(**inputs)

    # 5 logits -> 5 probabilities (index 0 = 1 star ... index 4 = 5 stars)
    probs = torch.softmax(outputs.logits, dim=1)[0].tolist()

    # Most likely star rating (1-5)
    stars = probs.index(max(probs)) + 1

    # 1-2 stars = negative, 3 = neutral, 4-5 = positive
    if stars < 3:
        sentiment = "NEGATIVE"
    elif stars == 3:
        sentiment = "NEUTRAL"
    else:
        sentiment = "POSITIVE"

    return sentiment, max(probs), stars, probs


@app.route("/", methods=["GET", "POST"])
def home():
    ctx = {"text": "", "error": None}

    if request.method == "POST":
        text = (request.form.get("user_input") or "").strip()
        ctx["text"] = text

        if not text:
            ctx["error"] = "Enter some text to analyze."
        else:
            sentiment, confidence, stars, probs = analyze_sentiment(text)
            ctx.update(
                sentiment=sentiment,
                confidence=confidence,
                stars=stars,
                probs=probs,
            )

    return render_template("index.html", **ctx)


if __name__ == "__main__":
    app.run(debug=True)