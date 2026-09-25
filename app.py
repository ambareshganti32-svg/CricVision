from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# =========================================================
# LOAD TRAINED MODEL
# =========================================================

try:
    model = joblib.load("cricket_model.pkl")
    encoders = joblib.load("encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    print("Model loaded successfully!")

except Exception as e:
    print("ERROR LOADING MODEL:")
    print(e)
    model = None
    encoders = None
    feature_columns = None


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# GET TEAM NAMES
# =========================================================

@app.route("/teams")
def teams():

    if encoders is None:
        return jsonify({
            "error": "Encoders are not loaded."
        }), 500

    try:

        # Get team names from the trained encoders
        team1_teams = list(encoders["Team1"].classes_)
        team2_teams = list(encoders["Team2"].classes_)

        # Combine and remove duplicates
        teams = sorted(
            list(set(team1_teams + team2_teams))
        )

        return jsonify({
            "teams": teams
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "error": "ML model is not loaded."
        }), 500

    try:

        data = request.get_json()

        # -------------------------------------------------
        # GET INPUT VALUES
        # -------------------------------------------------

        team1 = data.get("Team1", "").strip()
        team2 = data.get("Team2", "").strip()
        toss_winner = data.get("TossWinner", "").strip()
        toss_decision = data.get("TossDecision", "").strip()

        team1_score = float(data.get("Team1Score"))
        team2_score = float(data.get("Team2Score"))
        overs = float(data.get("Overs"))
        wickets = float(data.get("Wickets"))

        # -------------------------------------------------
        # BASIC VALIDATION
        # -------------------------------------------------

        if not team1:
            return jsonify({
                "error": "Please select Team 1."
            }), 400

        if not team2:
            return jsonify({
                "error": "Please select Team 2."
            }), 400

        if team1 == team2:
            return jsonify({
                "error": "Team 1 and Team 2 cannot be the same."
            }), 400

        if not toss_winner:
            return jsonify({
                "error": "Please select the toss winner."
            }), 400

        if toss_winner not in [team1, team2]:
            return jsonify({
                "error": "Toss winner must be Team 1 or Team 2."
            }), 400

        if toss_decision not in ["Bat", "Field"]:
            return jsonify({
                "error": "Invalid toss decision."
            }), 400

        if team1_score < 0 or team2_score < 0:
            return jsonify({
                "error": "Scores cannot be negative."
            }), 400

        if overs <= 0 or overs > 50:
            return jsonify({
                "error": "Overs must be between 0 and 50."
            }), 400

        if wickets < 0 or wickets > 10:
            return jsonify({
                "error": "Wickets must be between 0 and 10."
            }), 400

        # -------------------------------------------------
        # CHECK TEAM NAMES
        # -------------------------------------------------

        for col, value in [
            ("Team1", team1),
            ("Team2", team2),
            ("TossWinner", toss_winner)
        ]:

            if col not in encoders:
                return jsonify({
                    "error": f"{col} encoder is missing."
                }), 500

            known_values = list(
                encoders[col].classes_
            )

            if value not in known_values:

                return jsonify({
                    "error":
                    f"'{value}' is not available in the trained dataset."
                }), 400

        # -------------------------------------------------
        # CREATE INPUT DATAFRAME
        # -------------------------------------------------

        new_data = pd.DataFrame({

            "Team1": [team1],

            "Team2": [team2],

            "TossWinner": [toss_winner],

            "TossDecision": [toss_decision],

            "Team1Score": [team1_score],

            "Team2Score": [team2_score],

            "Overs": [overs],

            "Wickets": [wickets]
        })

        # -------------------------------------------------
        # ENCODE CATEGORICAL VALUES
        # -------------------------------------------------

        for col in [
            "Team1",
            "Team2",
            "TossWinner",
            "TossDecision"
        ]:

            new_data[col] = encoders[col].transform(
                new_data[col]
            )

        # -------------------------------------------------
        # SAME COLUMN ORDER AS TRAINING
        # -------------------------------------------------

        new_data = new_data[feature_columns]

        # -------------------------------------------------
        # MAKE PREDICTION
        # -------------------------------------------------

        prediction = model.predict(new_data)[0]

        # Your model uses:
        # Team1 = 1
        # Team2 = 0

        if prediction == 1:

            winner = team1

        else:

            winner = team2

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        result = {

            "winner": winner,

            "team1": team1,

            "team2": team2
        }

        # -------------------------------------------------
        # PREDICTION PROBABILITY
        # -------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                new_data
            )[0]

            classes = list(model.classes_)

            team1_probability = 0
            team2_probability = 0

            for class_value, probability in zip(
                classes,
                probabilities
            ):

                if class_value == 1:

                    team1_probability = (
                        float(probability) * 100
                    )

                elif class_value == 0:

                    team2_probability = (
                        float(probability) * 100
                    )

            result["team1_probability"] = round(
                team1_probability,
                2
            )

            result["team2_probability"] = round(
                team2_probability,
                2
            )

        return jsonify(result)

    # -----------------------------------------------------
    # ERROR HANDLING
    # -----------------------------------------------------

    except ValueError:

        return jsonify({
            "error":
            "Please enter valid numbers for score, overs and wickets."
        }), 400

    except KeyError as e:

        return jsonify({
            "error":
            f"Missing column or model information: {e}"
        }), 500

    except Exception as e:

        print("Prediction Error:")
        print(e)

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )