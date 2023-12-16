from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
import cv2
from PIL import Image
import os
from imageProcessing import returnDetectedFaces
import shutil
import base64
from io import BytesIO
from get_cnn_matches import get_cnn_embeddings, display_matches

trainingImagePath = "./uploaded_images/trainingDataSet/"
testingImagePath = "./uploaded_images/testingDataSet/"
LFWSampleImagePath = '/home/jorgejc2/Documents/ClassRepos/CS549FinalProjectFrontEnd/mtcnn_extracted_faces/Akbar_Hashemi_Rafsanjani/'
"/home/jorgejc2/Documents/ClassRepos/CS549FinalProjectFrontEnd/CS549Backend/uploaded_images/trainingDataSet"

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="http://localhost:4200")
CORS(app, 
    resources={r"/upload/*": {"origins": "http://localhost:4200"}, 
                     r"/get-training-images/*": {"origins": "http://localhost:4200"},
                     r"/get-testing-images/*": {"origins": "http://localhost:4200"},
                     r"/get-training-image-faces/*": {"origins": "http://localhost:4200"},
                     r"/get-testing-image-faces/*": {"origins": "http://localhost:4200"},
                     r"/get-sample-image-faces/*": {"origins": "http://localhost:4200"},
                     r"/flask_sockets/*": {"origins": "http://localhost:4200"},
                     r"/socket.io/*": {"origins": "http://localhost:4200"}
                     })

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

        # delete previous contents if folder if exists 
        shutil.rmtree(toSavePath, ignore_errors=True)
            
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
                # image = Image.open(os.path.join(toSavePath, img.filename))   
                # curr_img_embedding = get_cnn_embeddings(image)
                # display_matches(curr_img_embedding, image)

        

        # Process the uploaded file
        # You can save it, manipulate it, etc.
        # For this example, we'll return a success message
        return jsonify({'message': 'File uploaded successfully'})
    
    return render_template('upload.html')

@app.route('/get-training-images', methods=['GET'])
def get_training_images():
    start_index = request.args.get('startIndex', type=int, default=0)
    end_index = start_index + 20
    image_filenames = os.listdir(trainingImagePath)
    image_filenames = image_filenames[start_index:end_index]
    images = []
    for image_path in image_filenames:
        with open(trainingImagePath + image_path, 'rb') as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
            images.append(encoded_img)

    return jsonify(images)
     
@app.route('/get-testing-images', methods=['GET'])
def get_testing_images():
    start_index = request.args.get('startIndex', type=int, default=0)
    end_index = start_index + 20
    image_filenames = os.listdir(testingImagePath)
    image_filenames = image_filenames[start_index:end_index]
    images = []
    for image_path in image_filenames:
        with open(testingImagePath + image_path, 'rb') as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
            images.append(encoded_img)

    return jsonify(images)

@app.route('/get-training-image-faces', methods=['GET'])
def get_training_image_faces():
    start_index = request.args.get('startIndex', type=int, default=0)
    image_filenames = os.listdir(trainingImagePath)
    image_filename = image_filenames[start_index]


    face_images = returnDetectedFaces(trainingImagePath + image_filename)
    final_images = []

    for image in face_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
        final_images.append(encoded_img)

    return jsonify(final_images)
     
@app.route('/get-testing-image-faces', methods=['GET'])
def get_testing_image_faces():
    start_index = request.args.get('startIndex', type=int, default=0)
    image_filenames = os.listdir(testingImagePath)
    image_filename = image_filenames[start_index]


    face_images = returnDetectedFaces(testingImagePath + image_filename)
    final_images = []

    for image in face_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
        final_images.append(encoded_img)

    return jsonify(final_images)

@app.route('/get-sample-image-faces', methods=['GET'])
def get_sample_images():
    start_index = request.args.get('startIndex', type=int, default=0)
    end_index = start_index + 3
    image_filenames = os.listdir(LFWSampleImagePath)
    image_filenames = image_filenames[start_index:end_index]
    images = []
    for image_path in image_filenames:
        # Image.open(LFWSampleImagePath + image_path).show()
        with open(LFWSampleImagePath + image_path, 'rb') as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
            images.append(encoded_img)

    return jsonify(images)


def allowed_file(filename):
    # Implement a function to check the allowed file extensions
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif'}

# functions for debugging
@socketio.on('delayed_request')
def handle_delayed_request():
    # Simulate a delay of 5 seconds asynchronously
    print("Entered delayed response function")
    socketio.sleep(5)
    socketio.emit('delayed_response', {'message': 'Started Training your model'})
    print("Emitted delayed response")
    socketio.sleep(5)
    socketio.emit('delayed_response', {'message': 'Finished Training your model'})
    print("Emitted delayed response AGAIN")

if __name__ == '__main__':
    # app.run(
    #     host='0.0.0.0', port=5000,
    #     ssl_context=('server.crt', 'server.key'),
    #     debug=True
    # )
    socketio.run(app,
        host='0.0.0.0', port=5000,
        ssl_context=('server.crt', 'server.key'),
        debug=True
    )
        

