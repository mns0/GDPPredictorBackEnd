import time
import numpy as np
from app.helper import make_prediction
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for ,jsonify
    )

bp = Blueprint('api', __name__, url_prefix='/api')

@bp.route('/hello')
def hello():
    return 'Hello, World!'

@bp.route('/time')
def get_current_time():
    return {'time': time.time()}

@bp.route('/predict', methods=['GET'])
def predict():
    if request.method == 'GET':
        dates, gdp = make_prediction()
        return jsonify({
            'dates': dates.tolist(),
            'gdp': gdp.tolist()
            })

    
