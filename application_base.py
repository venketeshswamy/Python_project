#INSTALLED THE REQUIRED MODULES
import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.core.window import Window # It is or making my app background white
from kivy.uix.screenmanager import ScreenManager, Screen
lf=Builder.load_file("MAKE_HTML.kv")




#The following class is for Screen_1 Which z for selecting the template...
class Template_Screen (Screen):
    pass
class Builder_Screen (Screen):
    pass
class Template_1 (Screen):
    pass
class Template_2 (Screen):
    pass
class About_Screen (Screen):
    pass

#The Following class is for the Main App


class MAKE_HTML(App):
    """
    Still i need to write it fully.... Thanks for patience.....
    """
    def build(self):
        sm = ScreenManager()
        sm.add_widget(Template_Screen(name='Select Template'))
        sm.add_widget(Builder_Screen(name='Builder for customs'))
        sm.add_widget(Template_1(name='TEMP_1'))
        sm.add_widget(Template_2(name='TEMP_2'))
        sm.add_widget(About_Screen(name='About'))
        return sm
# calling the class for running the app
MAKE_HTML().run()
