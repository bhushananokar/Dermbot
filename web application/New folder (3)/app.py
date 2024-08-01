import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
model_filename = r"C:\Users\Bhushan\dermbot2.h5"
model_path = os.path.join(os.getcwd(), model_filename)
trained_model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = [
   "Acne and rosacea\nSymptoms: Stinging and burning of your skin . Red or pus-filled bumps that may resemble pimples . Patches of rough, dry skin.\nPrecautions: Protect your skin from the sun .Minimize stress . Protect your face from wind and cold." ,
   "Actinic keratosis\nSymptoms: Rough, dry or scaly patch of skin, usually less than 1 inch (2.5 centimeters) in diameter , Flat to slightly raised patch or bump on the top layer of skin , In some cases, a hard, wart like surface.\nPrecautions: Wear sunscreen every day , Use sunscreen daily , Avoid peak sun hours.",
   "Atopic dermatitis\nSymptoms: Dry, cracked skin , Itchiness , Rash on swollen skin that varies in color depending on your skin color.\nPrecautions: Keep water contact as brief as possible and use gentle body washes and cleansers instead of regular soaps. Lukewarm baths are better than long, hot baths , Do not scrub or dry the skin too hard or for too long , After bathing, apply lubricating ointments to damp skin.", 
   "Bullous disease\nSymptoms: Itching skin, weeks or months before blisters form , Large blisters that don't easily rupture when touched, often along creases or folds in the skin , Skin around the blisters that is normal, reddish or darker than normal.\nPrecautions: Avoid sun exposure , Avoid prolonged sun exposure on any area of the skin affected by bullous pemphigoid , Dress in loose fitting cotton clothes.",
   "Cellulitis impetigo\nSymptoms: Fever with chills and sweating , Skin redness or inflammation that gets bigger as the infection spreads , Skin sore or rash that starts suddenly, and grows quickly in the first 24 hours.\nPrecautions: Wash the area with clean water 2 times a day , Don't use hydrogen peroxide or alcohol, which can slow healing , You may cover the area with a thin layer of petroleum jelly, such as Vaseline, and a non-stick bandage.",
   "Eczema\nSymptoms: dry , cracked skin , thickened skin , itchiness , oozing.\nPrecautions: moisturize your skin often , avoid humidity , get regular exercise , manage stress.", 
   "Exanthems and drug eruption\nSymptoms: small , raised or flat , lesions on reddened skin.\nPrecautions: oral administration of antihistamines , moisturizing lotion.",
   "Alopecia (hair loss)\nSymptoms: gradual thinning on the top , circular and patchy bald spots , sudden loosening of hair.\nPrecautions: eat healthy diet that includes enough calories, proteins, iron, Find ways to cope with stress , manage thyroid disease", 
   "Herpes, HPV, or STDs\nSymptoms: Pain and itching around the genitals , painful urination , painful genital ulcers\nPrecautions: Use protection during copulation , avoid sexual contact with unknown and multiple partners.", 
   "Light diseases\nSymptoms: Pain in the eye , nausea or dizziness , numbness, score or wound in the eye.\nPrecautions: Wear a wide brim hat to the shade your face, head, ears, neck , Stay in shade avoid sunburn.",
    "Lupus \n Symptoms :Fatigue,ferver,butterfly shaped rash on face , skin lesions\nPrecautions:Avoid sunlight , use a spf50+ sunscreen",
    "Melanoma skin cancer\nSymptoms:moles,scaly patches,open sores or raised bumps", 
    "Nail fungus\nSymptoms:Thickened,discoloured,brittle,crumbly nails\nPrecautions:keep hands and feet dry and cleen", 
    "Poison ivy\nSymptoms:itching,rash,fluid filled blisters\nPrecaution:None . Vist doctor",
    "Psoriasis\nSymptoms:dry skin lesions, known as plaques, covered in scales.\nPrecaution:use moisturiziers,use humidifiers,use sunscreen",
    "Scabies, Lyme\nSymptoms: fever , fatigue , characteristic skin rash called as erythema migrans\nnPrecautions :wear long sleeved shirts and cloed shoes when in tick infested areas.",
    "Seborrheic\nSymptoms : dandruff or itching , rashes , flakiness , peeling or redness\nPrecautions : don’t use styling products , soften and remove scales from your hair", 
    "Systemic disease\nSymptoms : nausea , vomiting and abnormal liver function tests , infection in the lungs , urinary tract , skin or gastrointestinal tract\nPrecautions : use of analgesics and anti inflammatory agents",
    "Tinea ringworm candidiasis\nSymptoms : scaly patches, maybe red and itchy , fissures and darkening of the skin\nPrecautions : keep your child’s skin clean and dry , wear clean clothes and change socks and underwear each day", 
    "Urticaria (hives)\nSymptoms : skin rash due certain foods , medication and stress\nPrecautions : avoid triggers , take cold showers , protect your skin from sun",
    "Vascular tumors\nSymptoms: An area of spidery veins or lightened or discolored skin may appear before the hemangioma does , Lesions that grow under the skin in the fat may appear blue or purple , There may be no signs that hemangiomas have formed on an organ.\nPrecautions: Regular monitoring and imaging studies , Avoid trauma to the affected area , Use sun protection if the tumor is on the skin.", 
    "Vasculitis\nSymptoms: Trouble breathing , Numbness or tingling, especially in your hands and feet , Rashes, bumps or areas of discoloration on your skin.\nPrecautions: Consult a specialist for diagnosis and treatment , Take prescribed medications as directed , Regularly monitor your condition , Protect your skin from the sun.", 
    "Wart, molluscum\nSymptoms: Raised, round, skin-colored bumps , Small bumps — typically under about 1/4 inch (smaller than 6 millimeters) in diameter , Bumps with a small dent or dot at the top near the center.\nPrecautions: Wash your hands. Keeping your hands clean can help prevent spreading the virus , Avoid touching the bumps. Shaving over the infected areas also can spread the virus , Avoid sexual contact , Don't share or borrow personal items."
]

class SkinConditionClassifier:
    def predict_skin_condition(self, image_path):
        try:
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            predictions = trained_model.predict(img_array)
            predicted_class = np.argmax(predictions)

            if 0 <= predicted_class < len(class_labels):
                return class_labels[predicted_class]
            else:
                return "Error: Unrecognized class"

        except Exception as e:
            return f"Error: {str(e)}"

skin_condition_classifier = SkinConditionClassifier()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify_image', methods=['POST'])
def classify_image():
    try:
        # Retrieve the uploaded image from the HTML form
        uploaded_file = request.files['image']

        # Check if a file was uploaded
        if uploaded_file.filename != '':
            # Save the uploaded image temporarily
            image_path = 'uploaded_image.jpg'
            uploaded_file.save(image_path)

            # Perform classification
            predicted_class = skin_condition_classifier.predict_skin_condition(image_path)

            # Return the predicted class as JSON
            return jsonify({"result": predicted_class})

        else:
            return jsonify({"error": "No file uploaded"})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
