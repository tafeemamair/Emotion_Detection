import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys
from pathlib import Path

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from PIL import Image, ImageTk
import numpy as np
import cv2

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# Configuration
CONFIG = {
    "MODEL_WEIGHTS": SCRIPT_DIR / "model_weights1.h5",
    "CASCADE_FILE": SCRIPT_DIR / "haarcascade_frontalface_default.xml",
    "IMG_SIZE": 48,
    "EMOTIONS": ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"],
    "CONFIDENCE_THRESHOLD": 0.3,
}


def validate_file_exists(file_path, file_type):
    """Validate that a required file exists."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_type} not found at: {file_path}")
    return True


def Convolution(input_tensor, filters, kernel_size):
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same"
    )(input_tensor)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    return x


def Dense_f(input_tensor, nodes):
    x = Dense(nodes)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.25)(x)

    return x


def FacialExpressionModel(weights_file):
    try:
        validate_file_exists(weights_file, "Model weights file")

        inputs = Input((48, 48, 1))

        # Convolution Block 1
        x = Conv2D(
            64,
            (3, 3),
            padding="same"
        )(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # Convolution Block 2
        x = Conv2D(
            128,
            (5, 5),
            padding="same"
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # Convolution Block 3
        x = Conv2D(
            512,
            (3, 3),
            padding="same"
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # Flatten
        x = Flatten()(x)

        # Dense Block 1
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.25)(x)

        # Dense Block 2
        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.25)(x)

        # Output Layer
        output = Dense(
            7,
            activation="softmax"
        )(x)

        model = Model(
            inputs=inputs,
            outputs=output
        )

        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        model.load_weights(weights_file)

        return model

    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")


def initialize_cascade_classifier(cascade_path):
    """Load and validate Haar Cascade classifier."""
    try:
        validate_file_exists(cascade_path, "Cascade classifier file")
        
        facec = cv2.CascadeClassifier(str(cascade_path))
        if facec.empty():
            raise ValueError(f"Failed to load cascade classifier from: {cascade_path}")
        return facec
    except FileNotFoundError as e:
        raise RuntimeError(f"Cascade file error: {e}")
    except Exception as e:
        raise RuntimeError(f"Error initializing cascade classifier: {e}")


# Initialize GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = tk.Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = tk.Label(top)

# Initialize model and cascade with error handling
try:
    facec = initialize_cascade_classifier(CONFIG["CASCADE_FILE"])
    model = FacialExpressionModel(str(CONFIG["MODEL_WEIGHTS"]))
    print("✓ Model and cascade classifier loaded successfully")
except RuntimeError as e:
    print(f"✗ Initialization error: {e}")
    messagebox.showerror("Initialization Error", f"Failed to initialize:\n{e}")
    top.destroy()
    sys.exit(1)


def Detect(file_path):
    """Detect emotion from uploaded image file."""
    try:
        # Handle file path from different sources
        if hasattr(file_path, 'name'):
            # File object from filedialog
            file_path_str = file_path.name
        else:
            # String path
            file_path_str = str(file_path)
        
        # Validate file exists
        if not os.path.exists(file_path_str):
            label1.configure(foreground='#011638', text="File not found")
            print(f"✗ File not found: {file_path_str}")
            return
        
        # Read image
        image = cv2.imread(file_path_str)
        if image is None:
            label1.configure(foreground='#011638', text="Cannot read image file")
            print(f"✗ Unable to read image: {file_path_str}")
            return
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = facec.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(40, 40))
        
        if len(faces) == 0:
            label1.configure(foreground='#011638', text="No face detected")
            print("✗ No faces detected in image")
            return
        
        # Process each face and collect predictions
        predictions_list = []
        for (x, y, w, h) in faces:
            # Extract face region
            fc = gray_image[y:y+h, x:x+w]
            
            if fc.size == 0:
                print("⚠ Empty face crop, skipping")
                continue
            
            # Resize to model input size
            roi = cv2.resize(fc, (CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]))
            
            # Prepare input (add batch and channel dimensions)
            roi_input = roi[np.newaxis, :, :, np.newaxis].astype(np.float32)
            
            # Get predictions
            emotion_probs = model.predict(roi_input, verbose=0)
            pred_idx = np.argmax(emotion_probs[0])
            confidence = float(emotion_probs[0][pred_idx])
            
            emotion = CONFIG["EMOTIONS"][pred_idx]
            predictions_list.append((emotion, confidence))
            print(f"  Detected: {emotion} (confidence: {confidence:.2%})")
        
        if not predictions_list:
            label1.configure(foreground='#011638', text="No valid faces detected")
            print("✗ No valid faces processed")
            return
        
        # Use the highest confidence prediction
        best_emotion, best_confidence = max(predictions_list, key=lambda x: x[1])
        
        # Display result with confidence
        if best_confidence >= CONFIG["CONFIDENCE_THRESHOLD"]:
            result_text = f"{best_emotion}\n({best_confidence:.1%})"
            print(f"✓ Predicted Emotion: {best_emotion} ({best_confidence:.1%})")
        else:
            result_text = f"Uncertain\n({best_confidence:.1%})"
            print(f"⚠ Low confidence prediction: {best_emotion} ({best_confidence:.1%})")
        
        label1.configure(foreground="#011638", text=result_text)
    
    except Exception as e:
        error_msg = f"Error during emotion detection: {type(e).__name__}: {e}"
        print(f"✗ {error_msg}")
        label1.configure(foreground='#011638', text="Detection error")


def show_detect_button(file_path):
    """Create and display the Detect button."""
    detect_b = tk.Button(top, text="Detect Emotion", command=lambda: Detect(file_path),
                         padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)


def upload_image():
    """Handle image upload dialog."""
    try:
        file_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.bmp"),
        ("All files", "*.*")
    ]
)
        
        if not file_path:
            # User cancelled the dialog
            print("Upload cancelled by user")
            return
        
        # Try to open with PIL
        try:
            with Image.open(file_path) as img:
                uploaded = img.copy()
        except FileNotFoundError:
            label1.configure(foreground='#011638', text="File not found")
            print(f"✗ File not found: {file_path.name}")
            return
        except Exception as e:
            label1.configure(foreground='#011638', text=f"Cannot open: {type(e).__name__}")
            print(f"✗ Cannot open image: {type(e).__name__}: {e}")
            return
        
        # Resize for display
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='Image loaded. Click "Detect Emotion" to analyze.')
        
        # Show detect button with file path
        show_detect_button(file_path)
        print(f"✓ Image loaded: {file_path}")

    except Exception as e:
        error_msg = f"Unexpected error in upload_image: {type(e).__name__}: {e}"
        print(f"✗ {error_msg}")
        label1.configure(foreground='#011638', text="Upload error")


# Create UI elements
upload = tk.Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)

sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')

heading = tk.Label(top, text='Emotion Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

print("=" * 60)
print("Emotion Detection GUI Started")
print("=" * 60)

top.mainloop()
