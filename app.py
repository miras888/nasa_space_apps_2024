import os
from cs50 import SQL
from flask import Flask, flash, redirect, render_template, request, session, url_for, send_from_directory
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from helpers import apology, login_required
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import zscore

app = Flask(__name__)

# Configure session to use filesystem
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["PROCESSED_FOLDER"] = "processed"
Session(app)

# Ensure upload and processed folders exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["PROCESSED_FOLDER"], exist_ok=True)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///conquerors.db")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import zscore
import os
def detect_write(csv_file_directory):
    for csv_file in os.listdir(csv_file_directory):
        if csv_file.endswith('.csv'):
            path = f'{csv_file_directory}/{csv_file}'
            parts = csv_file.split('.')
            date_part = parts[4]
            year, month, day = date_part[:4], date_part[5:7], date_part[8:10]
            date = f'{year}-{month}-{day}'

            try:
                data_cat = pd.read_csv(path)
            except FileNotFoundError:
                print(f"File not found: {path}")
                continue

            csv_times = np.array(data_cat['time_rel(sec)'].tolist())
            csv_data = np.array(data_cat['velocity(m/s)'].tolist())
            z_scores = zscore(csv_data)

            threshold = 5
            anomalies = np.abs(z_scores) > threshold
            window_size = 1000
            num_anomalies_per_window = [np.sum(anomalies[i:i + window_size]) for i in range(0, len(csv_data), window_size)]
            max_anomalies_window_index = np.argmax(num_anomalies_per_window)
            start_idx = max_anomalies_window_index * window_size

            end_idx = None
            for i in range(start_idx, len(z_scores)):
                if np.abs(z_scores[i]) < 0.1:
                    end_idx = i
                    break

            if end_idx is not None:
                anomaly_duration = csv_times[end_idx] - csv_times[start_idx]
                anomaly_duration = round(anomaly_duration, 2)
                print(f"Duration of the anomaly: {anomaly_duration} seconds")

            # Plotting (optional visualization)
            fig, ax = plt.subplots(1, 1, figsize=(10, 3))
            ax.plot(csv_times, csv_data, label='Velocity (m/s)')
            ax.axvline(x=csv_times[start_idx], color='red', linestyle='--', label='Start of anomaly')
            ax.set_xlim([min(csv_times), max(csv_times)])
            ax.set_ylabel('Velocity (m/s)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'{csv_file}', fontweight='bold')
            ax.legend()

            # Debugging: print what will be inserted
            print(f"Inserting into database: Date: {date}, Path: {path}, Duration: {anomaly_duration}")

            # Inserting into the database
            db.execute("INSERT INTO moon_seismic_data (date, path, duration) VALUES (?, ?, ?)", (date, path, float(anomaly_duration)))




@app.route("/login", methods=["GET", "POST"])
def login():
    # Forget any user_id
    session.clear()

    if request.method == "POST":
        # Ensure username was submitted
        if not request.form.get("username"):
            return apology("must provide username", 403)

        # Ensure password was submitted
        elif not request.form.get("password"):
            return apology("must provide password", 403)

        # Query database for username
        rows = db.execute("SELECT * FROM users WHERE username = ?", request.form.get("username"))

        # Ensure username exists and password is correct
        if len(rows) != 1 or not check_password_hash(rows[0]["hash"], request.form.get("password")):
            return apology("invalid username and/or password", 403)

        # Remember which user has logged in
        session["user_id"] = rows[0]["id"]

        # Redirect user to home page
        return redirect("/")

    else:
        return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/login")


@app.route("/register", methods=["GET", "POST"])
def register():
    session.clear()

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")

        if not username:
            return apology("Type Username")
        elif not password:
            return apology("Type Password")
        elif not confirmation:
            return apology("Type Confirmation")
        elif password != confirmation:
            return apology("password don't match")

        rows = db.execute("SELECT * FROM users WHERE username = ?", username)

        if len(rows) > 0:
            return apology("Username already exists")

        db.execute("INSERT INTO users (username, hash) VALUES (?, ?)", username, generate_password_hash(password))
        rows1 = db.execute("SELECT * FROM users WHERE username = ?", request.form.get("username"))

        session["user_id"] = rows1[0]["id"]

        return redirect("/")

    else:
        return render_template("register.html")


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/mars", methods=["GET", "POST"])
@login_required
def mars():
    if request.method == "POST":
        # Retrieve filters
        year = request.form.get("year")
        month = request.form.get("month")
        day = request.form.get("day")

        # Build query with user-specified filters
        query = "SELECT * FROM mars_seismic_data WHERE 1=1"
        params = []

        if year:
            query += " AND strftime('%Y', date) = ?"
            params.append(year)
        if month:
            query += " AND strftime('%m', date) = ?"
            params.append(month)
        if day:
            query += " AND strftime('%d', date) = ?"
            params.append(day)

        # Execute query
        data = db.execute(query, *params)

        return render_template("mars.html", data=data, year=year, month=month, day=day)

    # When the page loads initially, no filter is applied
    data = db.execute("SELECT * FROM mars_seismic_data")
    return render_template("mars.html", data=data)


@app.route('/upload', methods=["POST"])
@login_required
def upload():
    if 'file' not in request.files:
        return apology("No file part", 400)
    
    file = request.files['file']

    if file.filename == '':
        return apology("No selected file", 400)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the file (placeholder for your algorithm)
        processed_file_path = os.path.join(app.config['PROCESSED_FOLDER'], "processed_" + filename)
        with open(file_path, 'r') as f:
            data = f.read()  # Replace this with actual processing logic

        # Simulate processing by writing the same data to processed folder
        with open(processed_file_path, 'w') as f:
            f.write(data)  # Replace this with processed data

        return redirect(url_for('download_processed', filename="processed_" + filename))


@app.route('/static/<path:filename>')
@login_required
def download_file(filename):
    # Serve the file from the static directory
    return send_from_directory('static', filename)


@app.route('/processed/<path:filename>')
@login_required
def download_processed(filename):
    # Serve the processed file from the processed directory
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route("/moon", methods=["GET", "POST"])
@login_required
def moon():
    if request.method == "POST":
        # Retrieve filters
        year = request.form.get("year")
        month = request.form.get("month")
        day = request.form.get("day")

        # Build query with user-specified filters
        query = "SELECT * FROM moon_seismic_data WHERE 1=1"
        params = []

        if year:
            query += " AND strftime('%Y', date) = ?"
            params.append(year)
        if month:
            query += " AND strftime('%m', date) = ?"
            params.append(month)
        if day:
            query += " AND strftime('%d', date) = ?"
            params.append(day)

        # Execute query
        data = db.execute(query, *params)

        return render_template("moon.html", data=data, year=year, month=month, day=day)

    # When the page loads initially, no filter is applied
    data = db.execute("SELECT * FROM moon_seismic_data")
    return render_template("moon.html", data=data)


@app.route('/upload_moon', methods=["POST"])
@login_required
def upload_moon():
    if 'file' not in request.files:
        return apology("No file part", 400)

    file = request.files['file']

    if file.filename == '':
        return apology("No selected file", 400)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the file (placeholder for your algorithm)
        processed_file_path = os.path.join(app.config['PROCESSED_FOLDER'], "processed_" + filename)
        with open(file_path, 'r') as f:
            data = f.read()  # Replace this with actual processing logic

        # Simulate processing by writing the same data to processed folder
        with open(processed_file_path, 'w') as f:
            f.write(data)  # Replace this with processed data

        return redirect(url_for('download_processed', filename="processed_" + filename))