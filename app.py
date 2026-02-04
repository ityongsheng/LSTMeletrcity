from flask import Flask, jsonify, request, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from service_layer import ShemsService
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'shems-secret-key-demo'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///shems.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

service = ShemsService()

# --- User Model ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(20), default='user') # user/admin

class Device(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    power_rating = db.Column(db.Float, nullable=False) # Watts
    priority = db.Column(db.Integer, default=1) # 1=High, 3=Low
    location = db.Column(db.String(50))
    status = db.Column(db.String(20), default='off') # on/off

class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    action = db.Column(db.String(100), nullable=False)
    details = db.Column(db.String(255))
    timestamp = db.Column(db.DateTime, default=db.func.now())
    ip_address = db.Column(db.String(50))

def log_audit(action, details=None):
    if current_user and current_user.is_authenticated:
        log = AuditLog(user_id=current_user.id, action=action, details=details, ip_address=request.remote_addr)
        db.session.add(log)
        db.session.commit()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Database Init ---
with app.app_context():
    db.create_all()

# --- Routes ---

@app.route('/')
@login_required
def index():
    return render_template('index.html', user=current_user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm = request.form.get('confirm_password')
        
        if password != confirm:
            flash('Passwords do not match.')
            return redirect(url_for('register'))
            
        if User.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('register'))
            
        new_user = User(username=username, password_hash=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        
        login_user(new_user)
        return redirect(url_for('index'))
        
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/api/simulation/step', methods=['POST'])
@login_required
def simulation_step():
    """
    Simulate one time step.
    For this demo, we just fetch a random or next sample from the loaded dataset.
    """
    # In a real app, 'current_data' would come from request params or iot stream
    # Ensure service is ready (might need re-init if hot reload messed it up, but singleton handles it)
    
    # Check if get_latest_data returns None
    result = service.get_latest_data()
    if not result:
        return jsonify({'error': 'Data not ready. Ensure model training is complete.'}), 500
        
    X_latest, y_actual = result
    
    # 1. Predict
    prediction = service.predict(X_latest) # X_latest is (1, T, F)
    
    # 2. Alert
    alert_status = service.alert(float(y_actual), prediction)
    
    # 3. Explain
    explanation = service.explain(X_latest)
    
    return jsonify({
        'timestamp': 'Current Step',
        'actual_power': float(y_actual),
        'prediction': prediction,
        'alert': alert_status,
        'explanation': explanation
    })

# --- New Modules Routes ---

@app.route('/devices')
@login_required
def devices_view():
    devices = Device.query.all()
    return render_template('devices.html', devices=devices)

@app.route('/api/devices', methods=['POST'])
@login_required
def add_device():
    name = request.form.get('name')
    power = request.form.get('power')
    
    dev = Device(name=name, power_rating=float(power))
    db.session.add(dev)
    db.session.commit()
    log_audit('Add Device', f'Added device {name}')
    return redirect(url_for('devices_view'))

@app.route('/api/devices/delete/<int:id>')
@login_required
def delete_device(id):
    dev = Device.query.get(id)
    if dev:
        db.session.delete(dev)
        db.session.commit()
        log_audit('Delete Device', f'Deleted device {dev.name}')
    return redirect(url_for('devices_view'))

@app.route('/logs')
@login_required
def logs_view():
    # Admin only check could go here
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(100).all()
    return render_template('logs.html', logs=logs)

@app.route('/reports')
@login_required
def reports_view():
    return render_template('reports.html')

@app.route('/simulation')
@login_required
def simulation_view():
    return render_template('simulation.html')

@app.route('/xai')
@login_required
def xai_view():
    return render_template('xai_lab.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
