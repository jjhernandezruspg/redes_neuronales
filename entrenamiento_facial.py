import os
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# ---------------- Configuración ----------------
IMG_SIZE = (48,48)
DISPLAY_SIZE = (600,600)
CLASSES = ['angry','disgust','fear','happy','neutral','sad','surprise']
MODEL_PATH = 'modelo_emociones.h5'
EPOCHS = 10
BATCH_SIZE = 32

camara_activa = False
current_batch = 0
batch_size = 4
image_files = []
esperada_actual = ""

# ---------------- Modelos ----------------
def crear_modelo_multiclase():
    model = Sequential([
        Conv2D(32,(3,3),activation='relu',input_shape=(48,48,1)),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128,(3,3),activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128,activation='relu'),
        Dropout(0.5),
        Dense(len(CLASSES),activation='softmax')
    ])
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def crear_modelo_binario():
    model = Sequential([
        Conv2D(32,(3,3),activation='relu',input_shape=(48,48,1)),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128,(3,3),activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128,activation='relu'),
        Dropout(0.5),
        Dense(1,activation='sigmoid')
    ])
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model

# ---------------- Callback progreso ----------------
class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_var, status_label):
        self.progress_var = progress_var
        self.status_label = status_label
    def on_epoch_end(self, epoch, logs=None):
        self.progress_var.set(int((epoch+1)/EPOCHS*100))
        self.status_label.config(text=f"Época {epoch+1}/{EPOCHS} | loss={logs['loss']:.4f} | acc={logs['accuracy']:.4f}")
        root.update_idletasks()

# ---------------- Entrenamiento ----------------
def entrenar_fer(progress_var,status_label):
    df = pd.read_csv('fer2013.csv')
    pixels = df['pixels'].tolist()
    images = np.array([np.fromstring(p,dtype=np.uint8,sep=' ').reshape(48,48) for p in pixels])
    images = np.expand_dims(images.astype('float32')/255.0,-1)
    labels = tf.keras.utils.to_categorical(df['emotion'].values,num_classes=len(CLASSES))
    X_train,X_val,y_train,y_val = train_test_split(images,labels,test_size=0.2,random_state=42)
    model = crear_modelo_multiclase()
    model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=EPOCHS,batch_size=BATCH_SIZE,
              callbacks=[ProgressCallback(progress_var,status_label)])
    model.save(MODEL_PATH)
    status_label.config(text="Entrenamiento FER2013 finalizado")

def entrenar_train(progress_var,status_label):
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory('train',target_size=IMG_SIZE,color_mode='grayscale',
                                            class_mode='categorical',batch_size=BATCH_SIZE)
    model = crear_modelo_multiclase()
    model.fit(train_gen,epochs=EPOCHS,callbacks=[ProgressCallback(progress_var,status_label)])
    model.save(MODEL_PATH)
    status_label.config(text="Entrenamiento train/ finalizado")

def entrenar_emocion_binaria(emocion,progress_var,status_label):
    carpeta = os.path.join('train',emocion)
    if not os.path.exists(carpeta):
        status_label.config(text=f"No existe carpeta {carpeta}")
        return
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory('train',target_size=IMG_SIZE,color_mode='grayscale',
                                            classes=[emocion],class_mode='binary',batch_size=BATCH_SIZE)
    model = crear_modelo_binario()
    model.fit(train_gen,epochs=EPOCHS,callbacks=[ProgressCallback(progress_var,status_label)])
    model.save(f"modelo_{emocion}.h5")
    status_label.config(text=f"Entrenamiento binario {emocion} finalizado")

# ---------------- Test por lotes ----------------
def mostrar_lote(carpeta, frame_imagenes, progress_var, status_label):
    global current_batch, batch_size, image_files, esperada_actual
    esperada_actual = os.path.basename(carpeta)
    # Limpiar visor de imágenes
    for widget in frame_imagenes.winfo_children():
        widget.destroy()
    image_files = [os.path.join(carpeta,f) for f in os.listdir(carpeta) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    current_batch = 0
    procesar_lote(frame_imagenes, progress_var, status_label)

def procesar_lote(frame_imagenes, progress_var, status_label):
    global current_batch, batch_size, image_files, esperada_actual
    if not os.path.exists(MODEL_PATH):
        status_label.config(text="Entrena primero el modelo")
        return
    model = load_model(MODEL_PATH)
    start = current_batch*batch_size
    end = min(start+batch_size,len(image_files))
    files_to_show = image_files[start:end]
    for widget in frame_imagenes.winfo_children():
        widget.destroy()
    col = 0
    for fpath in files_to_show:
        img = cv2.imread(fpath,cv2.IMREAD_GRAYSCALE)
        # Procesar imagen para modelo
        img_input = cv2.resize(img, IMG_SIZE).astype('float32')/255.0
        img_input = np.expand_dims(img_input, axis=-1)  # (48,48,1)
        img_input = np.expand_dims(img_input, axis=0)   # (1,48,48,1)
        pred = model.predict(img_input, verbose=0)
        clase = CLASSES[np.argmax(pred)]
        conf = np.max(pred)*100
        acierto = "✔" if clase==esperada_actual else "❌"
        # Mostrar imagen con nitidez
        img_disp = cv2.resize(img,DISPLAY_SIZE,interpolation=cv2.INTER_CUBIC)
        img_disp_rgb = cv2.cvtColor(img_disp,cv2.COLOR_GRAY2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img_disp_rgb))
        lbl = tk.Label(frame_imagenes,image=img_tk)
        lbl.image = img_tk
        lbl.grid(row=0,column=col,padx=5,pady=5)
        tk.Label(frame_imagenes,text=f"{clase} {conf:.1f}% {acierto}",bg='yellow').grid(row=1,column=col)
        col += 1
    progress_var.set(int(end/len(image_files)*100))
    status_label.config(text=f"Mostrando imágenes {start+1}-{end} de {len(image_files)}")

def siguiente_lote(frame_imagenes, progress_var, status_label):
    global current_batch, batch_size, image_files
    if (current_batch+1)*batch_size < len(image_files):
        current_batch +=1
        procesar_lote(frame_imagenes, progress_var, status_label)

def anterior_lote(frame_imagenes, progress_var, status_label):
    global current_batch
    if current_batch>0:
        current_batch -=1
        procesar_lote(frame_imagenes, progress_var, status_label)

# ---------------- Cámara ----------------
def prueba_camara(progress_var,status_label,image_label):
    global camara_activa
    if not os.path.exists(MODEL_PATH):
        status_label.config(text="Entrena primero el modelo")
        return

    model = load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_label.config(text="No se pudo abrir la cámara. Verifica permisos en macOS")
        return

    camara_activa = True

    # Limpiar visor
    for widget in frame_imagenes.winfo_children():
        widget.destroy()
    image_label.config(image='')

    def update_frame():
        global camara_activa
        if not camara_activa:
            cap.release()
            image_label.config(image='')
            return
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in faces:
                roi = gray[y:y+h,x:x+w]
                roi_resized = cv2.resize(roi, IMG_SIZE)
                roi_input = roi_resized.astype('float32')/255.0
                roi_input = np.expand_dims(roi_input, axis=-1)  # (48,48,1)
                roi_input = np.expand_dims(roi_input, axis=0)   # (1,48,48,1)
                pred = model.predict(roi_input, verbose=0)
                clase = CLASSES[np.argmax(pred)]
                conf = np.max(pred)*100
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,f"{clase} {conf:.1f}%",(x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),2,cv2.LINE_AA)
                progress_var.set(int(conf))
                status_label.config(text=f"Emoción: {clase} ({conf:.1f}%)")
            # Mejor nitidez y tamaño
            img_disp = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            img_disp = cv2.resize(img_disp,(DISPLAY_SIZE[0]+200,DISPLAY_SIZE[1]+200),interpolation=cv2.INTER_CUBIC)
            img_tk = ImageTk.PhotoImage(Image.fromarray(img_disp))
            image_label.config(image=img_tk)
            image_label.image = img_tk
        root.after(30, update_frame)
    update_frame()

def detener_camara():
    global camara_activa
    camara_activa = False

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Reconocimiento de Emociones")
root.geometry("1600x1000")
root.configure(bg="#222222")

frame_left = tk.Frame(root,bg="#333333",width=400)
frame_left.pack(side='left',fill='y')

frame_right = tk.Frame(root,bg="#444444")
frame_right.pack(side='right',fill='both',expand=True)

# Progress y status
progress_var = tk.IntVar()
progress_bar = ttk.Progressbar(frame_left,variable=progress_var,maximum=100)
progress_bar.pack(pady=10,fill='x',padx=5)
status_label = tk.Label(frame_left,text="Esperando acción",bg="#333333",fg="white")
status_label.pack(pady=5)

# Cámara
image_label = tk.Label(frame_right)
image_label.pack(pady=10)

# Panel imágenes test
frame_imagenes = tk.Frame(frame_right,bg="#222222")
frame_imagenes.pack(side='top',fill='both',expand=True)

# Botones entrenamiento FER2013
tk.Label(frame_left,text="Entrenamiento FER2013",bg="#333333",fg="white").pack(pady=5)
tk.Button(frame_left,text="Entrenar FER2013",
          command=lambda: threading.Thread(target=entrenar_fer,args=(progress_var,status_label)).start()).pack(fill='x',padx=5,pady=2)
tk.Button(frame_left,text="Entrenar TODO train",
          command=lambda: threading.Thread(target=entrenar_train,args=(progress_var,status_label)).start()).pack(fill='x',padx=5,pady=2)

# Botones entrenamiento binario
tk.Label(frame_left,text="Entrenamiento binario por emoción",bg="#333333",fg="white").pack(pady=5)
for e in CLASSES:
    tk.Button(frame_left,text=e,command=lambda emo=e: threading.Thread(target=entrenar_emocion_binaria,args=(emo,progress_var,status_label)).start()).pack(fill='x',padx=5,pady=1)

# Botones test por carpeta
tk.Label(frame_left,text="Test por carpeta",bg="#333333",fg="white").pack(pady=5)
for e in CLASSES:
    tk.Button(frame_left,text=e,command=lambda emo=e: mostrar_lote(os.path.join('test',emo),frame_imagenes,progress_var,status_label)).pack(fill='x',padx=5,pady=1)

# Botón cámara
tk.Button(frame_left,text="Probar en tiempo real",command=lambda: threading.Thread(target=prueba_camara,args=(progress_var,status_label,image_label)).start()).pack(fill='x',padx=5,pady=5)
tk.Button(frame_left,text="Detener cámara",command=detener_camara).pack(fill='x',padx=5,pady=2)

# Botones lote
tk.Button(frame_left,text="Siguiente lote",command=lambda: siguiente_lote(frame_imagenes,progress_var,status_label)).pack(fill='x',padx=5,pady=1)
tk.Button(frame_left,text="Anterior lote",command=lambda: anterior_lote(frame_imagenes,progress_var,status_label)).pack(fill='x',padx=5,pady=1)

root.mainloop()
