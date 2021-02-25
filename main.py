#Importing THE REQUIRED MODULES
import kivy
from kivy.uix.label import Label
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
lf=Builder.load_file("MAKE_HTML.kv")                    # Loading the Kivy file
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
#from kivy.uix.floatlayout import FloatLayout





# Following classes are for the popups which might prove helpful
class abtapp(BoxLayout):# It is for knowing about the app
    pass
class abttempscreen(BoxLayout):# It is for telling user about the screen which is for selecting templates
    pass
class abttemp2 (BoxLayout):# It is for telling user abt template 2 which is the construction template
    pass
class abttemp1(BoxLayout):# it is for telling user about the template one.
    pass
class abtbuilder(BoxLayout):# It is for telling user about the builder
    pass
#The following class is for Screen_1 Which z for selecting the template...
class Template_Screen (Screen):
    show=abtapp()
    About_APP_Popup=Popup(title="About",content=show)
    show=abttempscreen()
    About_Screen_Popup=Popup(title="Help Screen",content=show)
class Builder_Screen (Screen):
    show=abtbuilder()
    About_Screen_Popup=Popup(title="About",content=show)
class Template_1 (Screen):
    show=abttemp1()
    About_Screen_Popup=Popup(title="About",content=show)
class Template_2 (Screen):
    show=abttemp2()
    About_Screen_Popup=Popup(title="About",content=show)


    
#The Following class is for the Main App
class main(App):
    """
    Still i need to write it fully.... Thanks for patience..... At present, it has only screen managers
    """
    title="My App"
    def build(self):
        sm = ScreenManager()
        sm.add_widget(Template_Screen(name='Select Template'))
        sm.add_widget(Builder_Screen(name='Builder'))
        sm.add_widget(Template_1(name='T1'))
        sm.add_widget(Template_2(name='T2'))
        return sm

# calling the main class for running the app
main().run()
