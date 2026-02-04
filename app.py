from flask import Flask, jsonify, request, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from service_layer import ShemsService
import os

app = Flask(__name__)
# 配置应用密钥和数据库
app.config['SECRET_KEY'] = 'shems-secret-key-demo'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///shems.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # 设置未登录时的重定向页面

service = ShemsService()  # 实例化业务逻辑服务


# --- 用户模型 ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)  # 用户名
    password_hash = db.Column(db.String(150), nullable=False)  # 密码哈希值
    role = db.Column(db.String(20), default='user')  # 用户角色：user/admin


# --- 设备模型 ---
class Device(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)  # 设备名称
    power_rating = db.Column(db.Float, nullable=False)  # 额定功率（瓦特）
    priority = db.Column(db.Integer, default=1)  # 优先级：1=高，3=低
    location = db.Column(db.String(50))  # 设备位置
    status = db.Column(db.String(20), default='off')  # 状态：on/off


# --- 审计日志模型 ---
class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))  # 操作用户ID
    action = db.Column(db.String(100), nullable=False)  # 操作类型
    details = db.Column(db.String(255))  # 操作详情
    timestamp = db.Column(db.DateTime, default=db.func.now())  # 时间戳
    ip_address = db.Column(db.String(50))  # 用户IP地址


# --- 日志记录函数 ---
def log_audit(action, details=None):
    """记录用户操作到审计日志"""
    if current_user and current_user.is_authenticated:
        log = AuditLog(
            user_id=current_user.id,
            action=action,
            details=details,
            ip_address=request.remote_addr
        )
        db.session.add(log)
        db.session.commit()


# --- 用户加载器 ---
@login_manager.user_loader
def load_user(user_id):
    """Flask-Login所需的用户加载函数"""
    return User.query.get(int(user_id))


# --- 初始化数据库 ---
with app.app_context():
    db.create_all()  # 创建所有数据库表


# --- 路由定义 ---

@app.route('/')
@login_required  # 需要登录才能访问
def index():
    """首页：显示主控制面板"""
    return render_template('index.html', user=current_user)


@app.route('/login', methods=['GET', 'POST'])
def login():
    """用户登录页面"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        # 验证用户名和密码
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('用户名或密码错误。')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """用户注册页面"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm = request.form.get('confirm_password')

        # 验证密码一致性
        if password != confirm:
            flash('两次输入的密码不一致。')
            return redirect(url_for('register'))

        # 检查用户名是否已存在
        if User.query.filter_by(username=username).first():
            flash('用户名已存在。')
            return redirect(url_for('register'))

        # 创建新用户
        new_user = User(
            username=username,
            password_hash=generate_password_hash(password)
        )
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        return redirect(url_for('index'))

    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    """用户登出"""
    logout_user()
    return redirect(url_for('login'))


@app.route('/api/simulation/step', methods=['POST'])
@login_required
def simulation_step():
    """
    模拟一个时间步长。
    在实际应用中，'current_data' 应来自请求参数或物联网流
    这里从已加载的数据集中获取随机或下一个样本
    """
    # 获取最新数据用于模拟
    result = service.get_latest_data()
    if not result:
        return jsonify({'error': '数据未就绪。请确保模型训练已完成。'}), 500

    X_latest, y_actual = result

    # 1. 进行能耗预测
    prediction = service.predict(X_latest)  # X_latest 形状为 (1, T, F)

    # 2. 异常检测与预警
    alert_status = service.alert(float(y_actual), prediction)

    # 3. 生成SHAP解释
    explanation = service.explain(X_latest)

    return jsonify({
        'timestamp': '当前时间步',
        'actual_power': float(y_actual),  # 实际功耗
        'prediction': prediction,  # 预测结果
        'alert': alert_status,  # 预警状态
        'explanation': explanation  # 解释信息
    })


# --- 新模块路由 ---

@app.route('/devices')
@login_required
def devices_view():
    """设备管理页面"""
    devices = Device.query.all()
    return render_template('devices.html', devices=devices)


@app.route('/api/devices', methods=['POST'])
@login_required
def add_device():
    """添加新设备"""
    name = request.form.get('name')
    power = request.form.get('power')

    dev = Device(name=name, power_rating=float(power))
    db.session.add(dev)
    db.session.commit()
    log_audit('添加设备', f'添加设备 {name}')
    return redirect(url_for('devices_view'))


@app.route('/api/devices/delete/<int:id>')
@login_required
def delete_device(id):
    """删除设备"""
    dev = Device.query.get(id)
    if dev:
        db.session.delete(dev)
        db.session.commit()
        log_audit('删除设备', f'删除设备 {dev.name}')
    return redirect(url_for('devices_view'))


@app.route('/logs')
@login_required
def logs_view():
    """安全日志查看页面"""
    # 这里可以添加管理员权限检查
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(100).all()
    return render_template('logs.html', logs=logs)


@app.route('/reports')
@login_required
def reports_view():
    """报告中心页面"""
    return render_template('reports.html')


@app.route('/simulation')
@login_required
def simulation_view():
    """模拟计算页面"""
    return render_template('simulation.html')


@app.route('/xai')
@login_required
def xai_view():
    """可解释AI实验室页面"""
    return render_template('xai_lab.html')


# --- 主程序入口 ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)