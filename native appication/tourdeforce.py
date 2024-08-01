import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QFrame
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QPainterPath
from PyQt5.QtCore import Qt, QSize, QRect, QPropertyAnimation, QRectF
from PIL import Image

current_directory = os.getcwd()
model_filename = "dermbot2.h5"
model_path = os.path.join(current_directory, model_filename)
trained_model = tf.keras.models.load_model(model_path)

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

class SkinConditionClassifier(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        # Set the main window properties
        self.setWindowTitle("Dermbot")
        self.setGeometry(200, 200, 1600, 1200)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        # Create a black background for the main window
        central_widget.setStyleSheet("background-color: black;")

        central_frame = QFrame()
        central_frame.setStyleSheet("background-color: transparent;")
        central_layout = QVBoxLayout()

        # Create a grey square with rounded corners to display the content
        content_widget = QWidget(self)
        content_widget.setStyleSheet("background-color: #333333; border-radius: 20px;")

        self.title_label = QLabel("Dermbot", self)
        title_font = QFont()
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: white;")
        central_layout.addWidget(self.title_label, alignment=Qt.AlignTop | Qt.AlignHCenter)

        self.image_label = QLabel(self)
        central_layout.addWidget(self.image_label, alignment=Qt.AlignHCenter)

        self.result_label = QLabel("", self)
        result_font = QFont()
        result_font.setBold(True)
        self.result_label.setFont(result_font)
        self.result_label.setStyleSheet("color: white;")
        central_layout.addWidget(self.result_label, alignment=Qt.AlignHCenter)

        central_layout.addStretch(1)  # Add spacing

        # Create a browse button
        self.browse_button = QPushButton("Browse Image", self)
        self.browse_button.clicked.connect(self.browse_image)
        self.browse_button.setStyleSheet(
            "QPushButton { font-size: 16px; padding: 10px 20px; border-radius: 10px; background-color: #007BFF; color: white; }"
            "QPushButton:hover { background-color: #0056b3; }"
        )
        central_layout.addWidget(self.browse_button, alignment=Qt.AlignHCenter)

        # Create a report button
        self.report_button = QPushButton("Report", self)
        self.report_button.clicked.connect(self.generate_report)
        self.report_button.setStyleSheet(
            "QPushButton { font-size: 16px; padding: 10px 20px; border-radius: 10px; background-color: #28a745; color: white; }"
            "QPushButton:hover { background-color: #218838; }"
        )
        central_layout.addWidget(self.report_button, alignment=Qt.AlignHCenter)

        central_layout.addStretch(1)  # Add spacing

        content_widget.setLayout(central_layout)
        central_frame.setLayout(QVBoxLayout())
        central_frame.layout().addWidget(content_widget)

        layout.addWidget(central_frame, alignment=Qt.AlignCenter)
        layout.addStretch(1)  # Add spacing

        central_widget.setLayout(layout)

        # Store the image path for generating the report
        self.image_path = None

    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.jpg *.jpeg *.png *.jfif)")
        if file_path:
            self.image_path = file_path
            self.result_label.setText("")  # Clear previous result
            self.display_image(file_path)

    def display_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((300, 300))
        img = img.convert('RGB')
        image_data = np.array(img)
        h, w, ch = image_data.shape
        bytes_per_line = 3 * w
        q_image = QImage(image_data.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Adding margins and rounded corners to the displayed image
        rounded_pixmap = self.rounded_pixmap(pixmap)
        self.image_label.setPixmap(rounded_pixmap)

    def rounded_pixmap(self, pixmap):
        # Create a new QPixmap and QPainter object for the rounded pixmap
        rounded_pixmap = QPixmap(pixmap.size())
        rounded_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(rounded_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        rounded_rect = QRectF(0, 0, pixmap.width(), pixmap.height())
        path = QPainterPath()
        path.addRoundedRect(rounded_rect, 20, 20)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pixmap)

        # End the painting process
        painter.end()

        return rounded_pixmap

    def generate_report(self):
        if self.image_path:
            predicted_class, match_percent = self.predict_skin_condition(self.image_path)
            result_text = f"Predicted Class:\n{predicted_class}\nMatch Percent: {match_percent:.2f}%"
            self.result_label.setText(result_text)

    def predict_skin_condition(self, image_path):
        try:
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            predictions = trained_model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            confidence_score = predictions[0][predicted_class_index] * 100

            if 0 <= predicted_class_index < len(class_labels):
                return class_labels[predicted_class_index], confidence_score
            else:
                return "Error: Unrecognized class", 0.0
        except Exception as e:
            return f"Error: {str(e)}", 0.0

if __name__ == '__main__':
    app = QApplication([])
    window = SkinConditionClassifier()
    window.show()
    app.exec_()
