from flask import Flask, request, jsonify
import os
import base64
import shutil
import sys
import importlib.util

app = Flask(__name__)

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def run_floorplan_detector(image_path, tmp_dir):
    """
    Run the floorplan detector by importing and executing the flooplan_detector.py file
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        detector_path = os.path.join(current_dir, "flooplan_detector.py")
        
        # Import the detector module
        spec = importlib.util.spec_from_file_location("flooplan_detector", detector_path)
        detector_module = importlib.util.module_from_spec(spec)
        
        # Execute the module to load all functions
        spec.loader.exec_module(detector_module)
        
        # Call the main processing function
        result = detector_module.main_floorplan_processing(image_path, tmp_dir)
        
        return True, "Floorplan detector executed successfully"
    except Exception as e:
        return False, f"Failed to execute floorplan detector: {str(e)}"

@app.route('/vertx', methods=['POST'])
def execute_floorplan_detector():
    image_base64 = request.json.get('image')

    if not image_base64:
        return jsonify({
            "status": "error",
            "message": "Please provide an image base64 string"
        }), 400
    
    if image_base64.startswith("data:image"):
        image_base64 = image_base64.split(",")[1]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(current_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Save the uploaded image
    image_path = os.path.join(tmp_dir, "floorplan.png")
    with open(image_path, "wb") as f:
        f.write(base64.b64decode(image_base64))

    # Run the floorplan detector
    success, message = run_floorplan_detector(image_path, tmp_dir)
 
    if success:
        # Read the output files
        import ast

        def read_file(filename):
            file_path = os.path.join(tmp_dir, filename)
            try:
                with open(file_path, "r") as f:
                    return ast.literal_eval(f.read())
            except Exception as e:
                return None

        doors = read_file("doors_vertices.txt")
        floor = read_file("floor_vertices.txt")
        walls = read_file("walls_vertices.txt")
        windows = read_file("windows_vertices.txt")
        canvas = read_file("canvas_vertices.txt")

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        return jsonify({
            "doors": doors,
            "floor": floor,
            "walls": walls,
            "windows": windows,
            "canvas": canvas
        })
    else:
        return jsonify({
            "status": "error",
            "message": message
        }), 500

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "success",
        "message": "Hello, World!",
        "version": "0.0.1"
    })

@app.route('/health/torch')
def health_torch():
    import sys, os
    info = {"python": sys.executable, "PATH": os.environ.get("PATH")}
    try:
        import torch
        info.update({"ok": True, "torch": torch.__version__})
    except Exception as e:
        info.update({"ok": False, "error": str(e)})
    return jsonify(info), (200 if info["ok"] else 500)

if __name__=='__main__':
    app.run(host="0.0.0.0", port=5050, debug=False, use_reloader=False)
    # app.run(debug=True, host='0.0.0.0', port=5050)