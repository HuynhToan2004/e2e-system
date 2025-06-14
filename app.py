from flask import Flask, render_template, request, redirect, session, url_for
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import re
def is_valid_password(password):
    if len(password) < 10 or len(password) > 12:
        return False
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'[a-z]', password):
        return False
    if not re.search(r'[0-9]', password):
        return False
    return True
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config["MONGO_URI"] = "mongodb://mongo:27017/plateApp"
mongo = PyMongo(app)

@app.route('/')
def index():
    if 'username' in session:
        return redirect("http://localhost:8070/")  # Redirect sang FastAPI
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    users = mongo.db.users
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm = request.form['confirm_password']

        if password != confirm:
            return render_template('signup.html', error="Passwords do not match")

        if not is_valid_password(password):
            return render_template('signup.html', error="Password must be 10-12 characters long, contain at least one uppercase letter, one lowercase letter, and one digit.")

        if users.find_one({'$or': [{'username': username}, {'email': email}]}):
            return render_template('signup.html', error="Username or email already exists")

        hashed_pw = generate_password_hash(password)
        users.insert_one({
            'username': username,
            'email': email,
            'password': hashed_pw
        })
        return redirect('/login')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    users = mongo.db.users
    if request.method == 'POST':
        login_input = request.form.get('login')
        password = request.form['password']

        user = users.find_one({'$or': [{'username': login_input}, {'email': login_input}]})

        if user and check_password_hash(user['password'], password):
            session['username'] = user['username']
            return redirect("http://localhost:8070/")
        else:
            return render_template('login.html', error="Invalid username/email or password")

    return render_template('login.html')


@app.route('/change-password', methods=['GET', 'POST'])
def change_password():
    users = mongo.db.users

    if request.method == 'POST':
        login_input = request.form.get('login')
        old_password = request.form.get('old_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        user = users.find_one({'$or': [{'username': login_input}, {'email': login_input}]})

        if not user:
            return render_template('change_password.html', error="User not found.")

        if not check_password_hash(user['password'], old_password):
            return render_template('change_password.html', error="Old password is incorrect.")

        if new_password != confirm_password:
            return render_template('change_password.html', error="Passwords do not match.")

        if not is_valid_password(new_password):
            return render_template('change_password.html', error="Password must be 10–12 characters long and contain uppercase, lowercase, and digits.")

        hashed = generate_password_hash(new_password)
        users.update_one({'_id': user['_id']}, {'$set': {'password': hashed}})

        return redirect(url_for('login'))

    return render_template('change_password.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9111, debug=True)


