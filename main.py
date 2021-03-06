try:
    Fh=open("sample.html","r")# Checks if the file exists or not
    Fh.close()
except:
    fh=open("sample.html","w")# Creates when file does not exist
    fh.close()
else:
    Fh=open("samplerec.html","w")#Creates new one 
    fh=open("sample.html","r+")
    Fh.write(fh.read())
    Fh.close()
    fh.close()#Closes the file
    fh=open("sample.html","w")#For overwriting data
    fh.close()# Closing file for further use
#Importing THE REQUIRED MODULES
import os
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.factory import Factory
lf=Builder.load_file("MAKE_HTML.kv")  # Loading the Kivy file
class Builder_Screen(Screen):
    pass
#The Following class is for the Main App
class main(App):
    
    """
    Still i need to write it fully.... Thanks for patience..... At present, it has only screen managers
    """
    
    title="Html Code assist"
    def build(self):
        """
Build for screen manager."""
        sm = ScreenManager()
        sm.add_widget(Builder_Screen(name='Builder'))
        return sm
    abspathtofile="Nothing till now"
    wd=os.getcwd()
    os=__import__("os")
    def copytolocal(self,source):
        import os
        import shutil
        print(wd)
        ext=source.split(".")[-1]
        fn=source.split(r"\\")[-1]
        shutil.copyfile(source,"User Content/abcsdasd.{}".format(ext,fn))
    def appendtofile(self,text):
        fh=open("sample.html","a")
        fh.write(text)
    def openop(self):
        import webbrowser
        webbrowser.open("file://{}/sample.html".format(os.getcwd()))

# calling the main class for running the app
main().run()
