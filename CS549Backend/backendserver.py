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

        if 'isTrainingData' not in request.form:
            response = jsonify({'error': 'No isTrainingData part'})
            response.status_code = 400  # Set an appropriate error status code
            return response
        
        if 'images' not in request.files:
            response = jsonify({'error': 'No file part'})
            response.status_code = 400  # Set an appropriate error status code
            return response
    
        # file = request.files['file']
        uploaded_images = request.files.getlist('images')
        isTrainingDataStr = request.form.get('isTrainingData')
        print("isTrainingDataStr is {}".format(isTrainingDataStr))
        if (isTrainingDataStr == 'true'):
            isTrainingData = True
            toSavePath = 'uploaded_images/' + 'trainingDataSet/'
        else: 
            isTrainingData = False
            toSavePath = 'uploaded_images/' + 'testingDataSet/'
            
        
        print("Received list of images in Flask; Uploading files to {} folder".format(toSavePath))

        # create a directory to hold the files if it does not exist already 
        if not os.path.exists('uploaded_images/'):
                os.makedirs('uploaded_images/')

        if not os.path.exists('uploaded_images/testingDataSet/'):
                os.makedirs('uploaded_images/testingDataSet/')

        if not os.path.exists('uploaded_images/trainingDataSet/'):
                os.makedirs('uploaded_images/trainingDataSet/')

        for img in uploaded_images:
            if img and allowed_file(img.filename):      
                img.save(os.path.join(toSavePath, img.filename))
                image = Image.open(os.path.join(toSavePath, img.filename))            

        # Process the uploaded file
        # You can save it, manipulate it, etc.
        # For this example, we'll return a success message
        return jsonify({'message': 'File uploaded successfully'})
    
    return render_template('upload.html')

def allowed_file(filename):
    # Implement a function to check the allowed file extensions
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif'}

if __name__ == '__main__':
        app.run(
        host='0.0.0.0', port=5000,
        ssl_context=('server.crt', 'server.key'),
        debug=True
    )
        

