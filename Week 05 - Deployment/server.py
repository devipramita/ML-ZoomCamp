from flask import Flask, request, jsonify
import pickle

app = Flask("churn")

@app.route("/predict_churn", methods=["POST", "GET"])
def predict_churn():

    with open("model1.bin", 'rb') as f_in:
        model = pickle.load(f_in)
    
    with open("dv.bin", 'rb') as f_in:
        dv = pickle.load(f_in)

    customer = request.get_json()
    X = dv.transform(customer)
    y_pred = model.predict_proba(X)[0,1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == "__main__":

    app.run(debug=True)