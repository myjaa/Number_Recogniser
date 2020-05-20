#the most stable version

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Rectangle,Line
from kivy.config import Config
from kivy.graphics import Color
from kivy.clock import Clock
from functools import partial
from kivy.core.window import Window
import numpy as np
import cv2
import tensorflow as tf
import os

model=tf.keras.models.load_model('mnist_classifier.h5')


# -------------------------------------------- GUI --------------------------------------------------

class Interface(BoxLayout):
    pos_factor = 10
    grid_size = 50

    def __init__(self, **kwargs):
        Window.size=(500,580)
        super(Interface, self).__init__(**kwargs)

        self.orientation = 'vertical'

        self.wid = Widget()
        self.second = BoxLayout()

        self.start = Button(text='Guess', size_hint=(.3, .17), pos_hint={'top': 1})
        self.start.bind(on_press=self.guess)
        self.second.add_widget(self.start)

        self.reset = Button(text='Clear', size_hint=(.3, .17), pos_hint={'top': 1})
        self.reset.bind(on_press=self.reset_grid)
        self.second.add_widget(self.reset)

        self.Guess=Label(text='My Guess',size_hint=(.3, .17), pos_hint={'top': 1})
        self.second.add_widget(self.Guess)

        self.create_canvas()

        self.corrected_row = {}
        for i in range(self.grid_size):
            self.corrected_row[i] = (self.grid_size - 1) - i

        self.add_widget(self.second)
        self.add_widget(self.wid)

    def guess(self,instance):

        Window.screenshot(name='sample.png')
        image=cv2.imread('sample0001.png')
        image=image[51:,:]

        # print(image.shape)
        img_predict = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_predict = cv2.resize(img_predict, (28, 28))
        img_predict = np.reshape(img_predict, (1, 28, 28, 1))
        # print(img_predict.shape)

        prediction=model.predict(img_predict)
        # print(prediction)

        guessed_number=np.argmax(prediction)
        print('Guess: ',guessed_number)

        self.second.remove_widget(self.Guess)
        guessed_number_string='My Guess: '+str(guessed_number)
        self.Guess = Label(text=guessed_number_string,size_hint=(.3, .17), pos_hint={'top': 1})
        self.second.add_widget(self.Guess)

        if os.path.exists('sample0001.png'):
            os.remove('sample0001.png')

    def create_canvas(self):
        with self.wid.canvas:
            Color(1, 1, 1, 1, mode='rgba')
            Rectangle(pos=(0,0), size=(500, 530))


    def reset_grid(self, instance):
        self.wid.canvas.clear()
        self.create_canvas()

    def on_touch_down(self, touch):
        super().on_touch_down(touch)
        try:
            # in kivy grid co-ordinates follow the format (col,row)
            #also it helps in keeping the drawing parts within the canvas
            row, col = self.corrected_row[touch.pos[1] // self.pos_factor], int(touch.pos[0] // self.pos_factor) # it's basically corrected_row[COL//self.pos_factor] , int(ROW) => ROW , COL [normal]
            # print(row, col)

            g_row = touch.pos[0] // self.pos_factor
            g_col = touch.pos[1] // self.pos_factor
            with self.wid.canvas:
                Color(0, 0, 0, 1, mode='rgba')
                Rectangle(pos=(g_row * self.pos_factor, g_col * self.pos_factor), size=(40, 40))
        except KeyError:
            pass

    def on_touch_move(self, touch):
        try:
            row,col=self.corrected_row[touch.pos[1] // self.pos_factor],int(touch.pos[0] // self.pos_factor)
            # print(row, col)

            g_row = touch.pos[0] // self.pos_factor
            g_col = touch.pos[1] // self.pos_factor
            with self.wid.canvas:
                Color(0, 0, 0, 1, mode='rgba')
                Rectangle(pos=(g_row * self.pos_factor, g_col * self.pos_factor), size=(40, 40))
        except:
            pass


interface = Interface()

class A_starApp(App):
    def build(self):
        return interface


# ----------------------------------------------------main------------------------------------------------------------------
if __name__ == "__main__":
    A_starApp().run()