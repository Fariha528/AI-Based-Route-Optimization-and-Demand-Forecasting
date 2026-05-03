from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import traceback
import os # Added for path handling

# Import the functions from our *new* model_utils.py
from model_utils import get_optimized_route_from_osrm, predict_demand

# --- Basic Setup ---
# Set the template folder to be relative to this script
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)
CORS(app)
logging.basicConfig(level=logging.INFO)

# --- Frontend Route ---
@app.route('/')
def home():
    """Serves the main index.html dashboard from the 'templates' folder."""
    return render_template('index.html')

# --- API Endpoints ---

@app.route('/api/optimize-route', methods=['POST'])
def optimize_route():
    """
    API endpoint for fast route optimization.
    """
    app.logger.info("Received a request at /api/optimize-route")
    try:
        data = request.json
        locations = data.get('locations')
        if not locations or len(locations) < 2:
            return jsonify({"error": "Please provide at least two locations."}), 400

        optimization_result = get_optimized_route_from_osrm(locations)
        if not optimization_result:
            return jsonify({"error": "Failed to get a response from the routing service."}), 500

        optimized_indices = optimization_result["optimized_indices"]
        optimized_route = [locations[i] for i in optimized_indices]

        response_data = {
           "optimized_route": optimized_route,
           "duration_minutes": optimization_result["duration"],
           "route_geometry": optimization_result["route_geometry"]
        }
        
        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Error during optimization: {traceback.format_exc()}")
        return jsonify({"error": "An internal server error occurred."}), 500


@app.route('/api/demand-forecast', methods=['POST'])
def forecast_demand():
    """
    API endpoint for predicting demand using the new LGBM model.
    """
    data = request.json
    lat, lng, datetime_str = data.get('lat'), data.get('lng'), data.get('datetime')
    if not all([lat, lng, datetime_str]):
        return jsonify({"error": "Missing required data."}), 400
    try:
        demand_score = predict_demand(lat, lng, datetime_str)
        return jsonify({"demand_score": demand_score})
    except Exception as e:
        app.logger.error(f"Error during forecast: {e}")
        return jsonify({"error": "Failed to get forecast."}), 500

if __name__ == '__main__':
    # Makes sure the app runs in debug mode for development
    app.run(debug=True, port=5000)

