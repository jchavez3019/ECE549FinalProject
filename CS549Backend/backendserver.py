from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import cv2
from PIL import Image
import os

app = Flask(__name__)
CORS(app, resources={r"/upload/*": {"origins": "http://localhost:4200"}})

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        
        if 'file' not in request.files:
            response = jsonify({'error': 'No file part'})
            response.status_code = 400  # Set an appropriate error status code
            return response
    
        file = request.files['file']
        
        if file.filename == '':
            response = jsonify({'error': 'No selected file'})
            response.status_code = 400  # Set an appropriate error status code
            return response
        
        if file:
            if not os.path.exists('uploaded_images/'):
                    os.makedirs('uploaded_images/')
            file.save(os.path.join('uploaded_images/', file.filename))
            image = Image.open(os.path.join('uploaded_images/', file.filename))

        # Process the uploaded file
        # You can save it, manipulate it, etc.
        # For this example, we'll return a success message
        return jsonify({'message': 'File uploaded successfully'})
    
    return render_template('upload.html')

if __name__ == '__main__':
        app.run(
        host='0.0.0.0', port=5000,
        ssl_context=('server.crt', 'server.key'),
        debug=True
    )
        

