from flask import Flask
from vgg_utils_withsave import *
from vgg_scratch import *
from tensorflow.keras.models import Model
import time
import os
import mtcnn

# Create the ShareServiceClient object which will be used to create a container client
app = Flask(__name__)
mnt_path = "/mnt/azfile/"
root_path = os.path.join(mnt_path, "application-data")
input_path = os.path.join(root_path, "input_faces")
verified_path = os.path.join(root_path, "verified_faces")
vgg_descriptor = None
detector = None


def initialize_model():
    global vgg_descriptor
    global detector
    model = define_model()
    vgg_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    detector = mtcnn.MTCNN()


@app.route('/')
def hello_world():
    return 'Hello, World!'


# Go through all directories and files and list their paths
@app.route('/list')
def list_files():
    start = time.time()
    file_list = dict()
    for person in os.listdir(verified_path):
        person_path = os.path.join(verified_path, person)
        if os.path.isdir(person_path):
            file_list[person] = []
            for file in os.listdir(person_path):
                file_list[person].append(os.path.join(person_path, file))
    file_list["execution_time"] = time.time() - start
    return file_list


@app.route('/write')
def write_txt_file():
    with open(mnt_path + str(time.time()) + "-test.txt", "w") as f:
        f.write("Hello World!")
    return "File written"


@app.route('/verify')
def predict():
    try:
        input_img_path = os.path.join(input_path, "input.jpg")
        input_embedding = get_embedding(input_img_path, detector, vgg_descriptor)
        if input_embedding is None:
            raise Exception("No face detected in input image")

        all_distance = {}
        for persons in os.listdir(verified_path):
            # print(persons)
            person_distance = []
            images = []
            for image in os.listdir(os.path.join(verified_path, persons)):
                full_img_path = os.path.join(verified_path, persons, image)
                if full_img_path[-3:] == "jpg":
                    images.append(full_img_path)
                # Get embeddings
            embeddings = get_embeddings(images, detector, vgg_descriptor)
            if embeddings is None:
                print("No faces detected")
                continue
            # Check if the input face is a match for the known face
            # print("input_embedding", input_embedding)
            for embedding in embeddings:
                score = is_match(embedding, input_embedding)
                person_distance.append(score)
            # Calculate the average distance for each person
            all_distance[persons] = np.mean(person_distance)
        top_ten = sorted(all_distance.items(), key=lambda x: x[1])[:10]
        return top_ten
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    initialize_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
